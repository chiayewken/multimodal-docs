import copy
import os
import time
import warnings
from typing import Optional, List, Union, Any

import google.generativeai as genai
import langdetect
import requests
import tiktoken
import torch
from PIL import Image
from anthropic import Anthropic
from dotenv import load_dotenv
from fire import Fire
from huggingface_hub import hf_hub_download
from langdetect import LangDetectException
from openai import OpenAI
from openai.lib.azure import AzureOpenAI
from pydantic import BaseModel
from qwen_vl_utils import process_vision_info
from reka.client import Reka
from transformers import (
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    Idefics2Processor,
    Idefics2ForConditionalGeneration,
    PreTrainedModel,
    ProcessorMixin,
    PreTrainedTokenizer,
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    BatchEncoding,
)

from data_loading import load_image_from_url, convert_image_to_text, MultimodalDocument
from onevision import (
    load_pretrained_model,
    process_images,
    tokenizer_image_token,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    conv_templates,
)

try:
    # noinspection PyUnresolvedReferences
    import lmdeploy
    from lmdeploy.serve.vl_async_engine import VLAsyncEngine
except ImportError:
    VLAsyncEngine = None


class DummyFastText:
    _FastText = None


class DummyClass:
    LLM = None
    SamplingParams = None
    FastText = DummyFastText


try:
    import vllm
except ImportError:
    vllm = DummyClass()
    print("Cannot import vllm, using DummyClass")


try:
    import fasttext
except ImportError:
    fasttext = DummyClass()
    print("Cannot import fasttext, using DummyClass")


def get_environment_key(path: str, name: str) -> str:
    assert os.path.exists(path), f"Path {path} does not exist"
    load_dotenv(path)
    return os.environ[name]


def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    if width > height:
        new_width = max_size
        new_height = round(height * max_size / width)
    else:
        new_height = max_size
        new_width = round(width * max_size / height)
    return image.resize((new_width, new_height), Image.LANCZOS)


class EvalModel(BaseModel, arbitrary_types_allowed=True):
    engine: str
    timeout: int = 60
    temperature: float = 0.0
    max_output_tokens: int = 512

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        raise NotImplementedError

    def run_many(self, inputs: List[Union[str, Image.Image]], num: int) -> List[str]:
        raise NotImplementedError


class GeminiModel(EvalModel):
    engine: str = "gemini-1.5-pro-001"
    client: Optional[genai.GenerativeModel] = None

    def load(self):
        if self.client is None:
            genai.configure(api_key=get_environment_key(".env", "GEMINI_KEY"))
            self.client = genai.GenerativeModel(self.engine)

    def run(self, inputs: Union[str, Image.Image]) -> str:
        self.load()
        output = ""
        config = genai.types.GenerationConfig(
            candidate_count=1,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        while not output:
            try:
                response = self.client.generate_content(
                    inputs, generation_config=config
                )
                if "block_reason" in str(vars(response)):
                    output = str(vars(response))
                elif not response.parts:
                    output = "Empty response.parts from gemini"
                else:
                    output = response.text
            except Exception as e:
                print(e)

            if not output:
                print("Model request failed, retrying.")
                time.sleep(1)

        return output


class GeminiFlashModel(GeminiModel):
    engine: str = "gemini-1.5-flash-001"


class OpenAIModel(EvalModel):
    engine: str = "gpt-4o-2024-05-13"
    client: Optional[OpenAI] = None

    def load(self):
        if self.client is None:
            key = get_environment_key(".env", "OPENAI_KEY")
            self.client = OpenAI(api_key=key, timeout=self.timeout)

    @staticmethod
    def make_messages(inputs: List[Union[str, Image.Image]]) -> List[dict]:
        outputs = []

        for x in inputs:
            if isinstance(x, str):
                outputs.append(dict(type="text", text=x))
            elif isinstance(x, Image.Image):
                outputs.append(
                    dict(
                        type="image_url",
                        image_url=dict(
                            url=f"data:image/png;base64,{convert_image_to_text(x)}"
                        ),
                    )
                )
            else:
                raise ValueError(f"Unsupported input type: {type(x)}")

        return [dict(role="user", content=outputs)]

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()
        output = ""
        error_message = "The response was filtered"

        while not output:
            try:
                response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=self.make_messages(inputs),
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                )
                if response.choices[0].finish_reason == "content_filter":
                    raise ValueError(error_message)
                output = response.choices[0].message.content

            except Exception as e:
                print(e)
                if error_message in str(e):
                    output = error_message

            if not output:
                print("OpenAIModel request failed, retrying.")

        return output


class OpenAIMiniModel(OpenAIModel):
    engine: str = "gpt-4o-mini-2024-07-18"


class AzureModel(OpenAIModel):
    engine: str = "gpt4o-0513"

    def load(self):
        if self.client is None:
            self.client = AzureOpenAI(
                azure_endpoint=get_environment_key(".env", "AZURE_ENDPOINT"),
                api_key=get_environment_key(".env", "AZURE_KEY"),
                api_version="2024-02-15-preview",
            )


class TextOnlyAzureModel(AzureModel):
    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        return super().run([x for x in inputs if isinstance(x, str)])


class AzureMiniModel(AzureModel):
    engine: str = "gpt-4o-mini"


class ClaudeModel(EvalModel):
    engine: str = "claude-3-5-sonnet-20240620"
    client: Optional[Anthropic] = None

    def load(self):
        if self.client is None:
            key = get_environment_key(".env", "CLAUDE_KEY")
            self.client = Anthropic(api_key=key, timeout=self.timeout)

    @staticmethod
    def make_messages(inputs: List[Union[str, Image.Image]]) -> List[dict]:
        outputs = []

        for x in inputs:
            if isinstance(x, str):
                outputs.append(dict(type="text", text=x))
            elif isinstance(x, Image.Image):
                data = dict(
                    type="image",
                    source=dict(
                        type="base64",
                        media_type="image/png",
                        data=convert_image_to_text(x),
                    ),
                )
                outputs.append(data)
            else:
                raise ValueError(f"Unsupported input type: {type(x)}")

        return [dict(role="user", content=outputs)]

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()
        output = ""
        error_message = "The response was filtered"

        while not output:
            try:
                response = self.client.messages.create(
                    model=self.engine,
                    messages=self.make_messages(inputs),
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                )
                output = response.content[0].text

            except Exception as e:
                print(e)
                if error_message in str(e):
                    output = error_message

            if not output:
                print("ClaudeModel request failed, retrying.")
        return output


class ClaudeHaikuModel(ClaudeModel):
    engine: str = "claude-3-haiku-20240307"


class RekaModel(EvalModel):
    engine: str = "reka-core-20240501"
    client: Optional[Reka] = None

    def load(self):
        if self.client is None:
            key = get_environment_key(".env", "REKA_KEY")
            self.client = Reka(api_key=key, timeout=self.timeout)

    @staticmethod
    def make_messages(inputs: List[Union[str, Image.Image]]) -> List[dict]:
        messages = []
        if len([x for x in inputs if isinstance(x, Image.Image)]) > 6:
            raise ValueError("RekaModel only supports up to 6 images.")

        for x in inputs:
            if isinstance(x, Image.Image):
                image_url = f"data:image/png;base64,{convert_image_to_text(x)}"
                content = [dict(type="image_url", image_url=image_url)]
                messages.append(dict(role="user", content=content))

        content = []
        for x in inputs:
            if isinstance(x, str):
                content.append(dict(type="text", text=x))
        if content:
            messages.append(dict(role="user", content=content))

        return messages

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()
        output = ""

        while not output:
            try:
                response = self.client.chat.create(
                    model=self.engine,
                    messages=self.make_messages(inputs),
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                )
                output = response.responses[0].message.content

            except Exception as e:
                print(e)
            if not output:
                print("RekaModel request failed, retrying.")

        return output


class InternModel(EvalModel):
    engine: str = "OpenGVLab/InternVL2-26B-AWQ"
    client: Optional[VLAsyncEngine] = None
    image_size: int = 448

    def load(self):
        if self.client is None:
            backend_config = lmdeploy.TurbomindEngineConfig(model_format="awq")
            # noinspection PyTestUnpassedFixture
            self.client = lmdeploy.pipeline(self.engine, backend_config=backend_config)

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()
        config = lmdeploy.GenerationConfig(max_new_tokens=self.max_output_tokens)
        text = "\n\n".join([x for x in inputs if isinstance(x, str)])
        size = self.image_size
        images = [resize_image(x, size) for x in inputs if isinstance(x, Image.Image)]
        response = self.client((text, images), generation_config=config)
        return response.text


class HighresInternModel(InternModel):
    image_size: int = 768


class OwlModel(EvalModel):
    engine: str = "mPLUG/mPLUG-Owl3-7B-240728"
    model: Optional[PreTrainedModel] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    processor: Optional[Any] = None

    def load(self):
        if self.model is None or self.tokenizer is None or self.processor is None:
            self.model = AutoModel.from_pretrained(
                self.engine,
                attn_implementation="sdpa",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            ).cuda()
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.engine, trust_remote_code=True
            )
            self.processor = self.model.init_processor(self.tokenizer)

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()

        text_inputs = [x for x in inputs if isinstance(x, str)]
        image_inputs = [x for x in inputs if isinstance(x, Image.Image)]

        messages = [
            {
                "role": "user",
                "content": "<|image|>" * len(image_inputs)
                + "\n"
                + "\n".join(text_inputs),
            },
            {"role": "assistant", "content": ""},
        ]

        processed_inputs = self.processor(messages, images=image_inputs, videos=None)
        processed_inputs.to("cuda")
        processed_inputs.update(
            {
                "tokenizer": self.tokenizer,
                "max_new_tokens": self.max_output_tokens,
                "decode_text": True,
            }
        )

        outputs = self.model.generate(**processed_inputs)
        return outputs[0]


class CogVLMModel(EvalModel):
    engine: str = "THUDM/cogvlm2-llama3-chat-19B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_type: torch.dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    tokenizer: Optional[PreTrainedTokenizer] = None
    model: Optional[PreTrainedModel] = None

    def load(self):
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.engine, trust_remote_code=True
            )
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.engine,
                    torch_dtype=self.torch_type,
                    trust_remote_code=True,
                )
                .to(self.device)
                .eval()
            )

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()

        text_inputs = [x for x in inputs if isinstance(x, str)]
        image_inputs = [x for x in inputs if isinstance(x, Image.Image)]

        query = "\n".join(text_inputs)
        images = [img.convert("RGB") for img in image_inputs]

        input_by_model = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=query,
            history=[],
            images=images if images else None,
            template_version="chat",
        )

        model_inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(self.device),
            "token_type_ids": input_by_model["token_type_ids"]
            .unsqueeze(0)
            .to(self.device),
            "attention_mask": input_by_model["attention_mask"]
            .unsqueeze(0)
            .to(self.device),
            "images": [
                [
                    img.to(self.device).to(self.torch_type)
                    for img in input_by_model["images"]
                ]
            ]
            if images
            else None,
        }

        gen_kwargs = {
            "max_new_tokens": self.max_output_tokens,
            "pad_token_id": 128002,
        }

        with torch.no_grad():
            outputs = self.model.generate(**model_inputs, **gen_kwargs)
            outputs = outputs[:, model_inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0]

        return response.strip()


class OneVisionModel(EvalModel):
    path: str = "models/onevision"
    engine: str = "lmms-lab/llava-onevision-qwen2-7b-ov"
    tokenizer: Optional[PreTrainedTokenizer] = None
    model: Optional[PreTrainedModel] = None
    processor: Optional[ProcessorMixin] = None
    image_size: int = 768

    def load(self):
        if self.model is None:
            path = self.path if os.path.exists(self.path) else self.engine
            print(dict(load_path=path))
            (
                self.tokenizer,
                self.model,
                self.processor,
                _,
            ) = load_pretrained_model(path, None, "llava_qwen")
            self.model.eval()

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()
        warnings.filterwarnings("ignore")
        size = self.image_size
        images = [resize_image(x, size) for x in inputs if isinstance(x, Image.Image)]
        if not images:
            images = [Image.new("RGB", (size, size), (255, 255, 255))]
        image_tensor = process_images(images, self.processor, self.model.config)
        image_list = [x.to(dtype=torch.float16, device="cuda") for x in image_tensor]

        conv_template = "qwen_1_5"
        text = "\n\n".join([x for x in inputs if isinstance(x, str)])
        question = DEFAULT_IMAGE_TOKEN + f"\n{text}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to("cuda")
        )
        image_sizes = [x.size for x in images]

        outputs = self.model.generate(
            input_ids,
            images=image_list,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=self.max_output_tokens,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


class HighresOneVisionModel(OneVisionModel):
    image_size: int = 768


class QwenModel(EvalModel):
    path: str = "models/qwen"
    engine: str = "Qwen/Qwen2-VL-7B-Instruct"
    model: Optional[Qwen2VLForConditionalGeneration] = None
    processor: Optional[Qwen2VLProcessor] = None
    device: str = "cuda"
    image_size: int = 768
    lora_path: str = ""

    def load(self):
        if self.model is None:
            path = self.path if os.path.exists(self.path) else self.engine
            print(dict(load_path=path))
            # noinspection PyTypeChecker
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                path, torch_dtype="auto", device_map="auto"
            )

            if self.lora_path:
                print("Loading LORA from", self.lora_path)
                self.model.load_adapter(self.lora_path)

            self.model = self.model.to(self.device).eval()
            self.processor = Qwen2VLProcessor.from_pretrained(self.engine)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)

    def make_messages(self, inputs: List[Union[str, Image.Image]]) -> List[dict]:
        text = "\n\n".join([x for x in inputs if isinstance(x, str)])
        content = [
            dict(
                type="image",
                image=f"data:image;base64,{convert_image_to_text(resize_image(x, self.image_size))}",
            )
            for x in inputs
            if isinstance(x, Image.Image)
        ]
        content.append(dict(type="text", text=text))
        return [dict(role="user", content=content)]

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()
        messages = self.make_messages(inputs)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        # noinspection PyTypeChecker
        model_inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=self.max_output_tokens
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]


class TextOnlyQwenModel(QwenModel):
    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        return super().run([x for x in inputs if isinstance(x, str)])


class HighresQwenModel(QwenModel):
    image_size: int = 1024


class CustomQwenModel(QwenModel):
    # engine: str = "models/qwen2_vl_lora_sft"
    lora_path: str = "saves/qwen2_vl-7b/lora/sft"

    # def load(self):
    #     super().load()
    #     # noinspection PyUnresolvedReferences
    #     template = self.processor.tokenizer.chat_template
    #     if template is not None:
    #         self.processor.chat_template = template


class GemmaModel(EvalModel):
    engine: str = "google/paligemma-3b-mix-448"
    model: Optional[PaliGemmaForConditionalGeneration] = None
    processor: Optional[PaliGemmaProcessor] = None
    device: str = "cuda"

    def load(self):
        if self.model is None:
            # noinspection PyTypeChecker
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                self.engine,
                torch_dtype=torch.bfloat16,
                revision="bfloat16",
            )
            self.model = self.model.to(self.device).eval()
            self.processor = PaliGemmaProcessor.from_pretrained(self.engine)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()
        prompt = " ".join([x for x in inputs if isinstance(x, str)])
        images = [x for x in inputs if isinstance(x, Image.Image)]
        raw = self.processor(text=prompt, images=images, return_tensors="pt")
        length = raw["input_ids"].shape[-1]

        with torch.inference_mode():
            # noinspection PyTypeChecker
            generation = self.model.generate(
                **raw.to(self.device),
                max_new_tokens=self.max_output_tokens,
                do_sample=True,  # Otherwise the outputs will be very repetitive
            )
            generation = generation[0][length:]
            return self.processor.decode(generation, skip_special_tokens=True)


class IdeficsModel(EvalModel):
    engine: str = "models/idefics"  # Optimized for long interleaved cases
    model: Optional[Idefics2ForConditionalGeneration] = None
    processor: Optional[Idefics2Processor] = None
    device: str = "cuda"
    image_size: int = 768

    def load(self):
        if self.model is None:
            # noinspection PyTypeChecker
            self.model = Idefics2ForConditionalGeneration.from_pretrained(self.engine)
            self.model = self.model.to(self.device).eval()
            self.processor = Idefics2Processor.from_pretrained(self.engine)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)

    def process_inputs(self, inputs: List[Union[str, Image.Image]]):
        self.load()
        text = "\n\n".join([x for x in inputs if isinstance(x, str)])
        content = [dict(type="image") for x in inputs if isinstance(x, Image.Image)]
        content.append(dict(type="text", text=text))

        messages = [dict(role="user", content=content)]
        size = self.image_size
        images = [resize_image(x, size) for x in inputs if isinstance(x, Image.Image)]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        return self.processor(
            text=prompt, images=images or None, return_tensors="pt"
        ).to(self.device)

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()
        with torch.inference_mode():
            outputs = self.model.generate(
                **self.process_inputs(inputs),
                max_new_tokens=self.max_output_tokens,
                do_sample=False,
            )
            texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
            return texts[0].split("Assistant:", maxsplit=1)[-1].strip()


class HighresIdeficsModel(IdeficsModel):
    image_size: int = 768


class CloudModel(EvalModel):
    # Queries the server on Google cloud for completions, which should work in any country
    url: str = "http://35.208.161.228:8000/completions"
    tokenizer: Optional[tiktoken.Encoding] = None
    costs: List[float] = []

    def get_model_key(self) -> str:
        if self.engine.startswith("gpt"):
            return get_environment_key(".env", "OPENAI_KEY")
        elif self.engine.startswith("claude"):
            return get_environment_key(".env", "CLAUDE_KEY")
        elif self.engine.startswith("gemini"):
            return get_environment_key(".env", "GEMINI_KEY")
        elif self.engine.startswith("reka"):
            return get_environment_key(".env", "REKA_KEY")
        else:
            raise ValueError(f"Unknown engine cannot find key: {self.engine}")

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        text = text.replace("<|endoftext|>", "")
        return len(self.tokenizer.encode(text))

    def count_cost(self, data: List[Union[str, Image.Image]], is_input: bool) -> float:
        # Estimate based on OpenAI gpt-4o-2024-08-06 and 1024x1024 images
        input_cost, output_cost, image_cost = (2.5 / 1e6, 10 / 1e6, 0.001913)
        texts = [x for x in data if isinstance(x, str)]
        images = [x for x in data if isinstance(x, Image.Image)]
        num_tokens = self.count_tokens(" ".join(texts))
        cost = num_tokens * (input_cost if is_input else output_cost)
        cost += len(images) * image_cost
        return cost

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        contents = [
            dict(text=x) if isinstance(x, str) else dict(image=convert_image_to_text(x))
            for x in inputs
        ]

        output = ""
        while not output:
            try:
                data = dict(
                    engine=self.engine,
                    key=self.get_model_key(),
                    contents=contents,
                    kwargs=dict(
                        timeout=self.timeout,
                        temperature=self.temperature,
                        max_tokens=self.max_output_tokens,
                    ),
                )

                headers = {"Content-Type": "application/json"}
                response = requests.post(self.url, json=data, headers=headers)
                if response.status_code == 200:
                    result = response.json()
                    output = result["texts"][0]
                else:
                    print("Error:", response.status_code, response.text)
            except Exception as e:
                print("CloudModel request failed:", e)
                time.sleep(1)

        cost = self.count_cost(inputs, is_input=True) + self.count_cost(
            [output], is_input=False
        )
        self.costs.append(cost)
        print(dict(cost=cost, average=sum(self.costs) / len(self.costs)))
        return output

    def run_many(self, inputs: List[Union[str, Image.Image]], num: int) -> List[str]:
        contents = [
            dict(text=x) if isinstance(x, str) else dict(image=convert_image_to_text(x))
            for x in inputs
        ]

        outputs = []
        while not outputs:
            try:
                data = dict(
                    engine=self.engine,
                    key=self.get_model_key(),
                    contents=contents,
                    num_generate=num,
                    kwargs=dict(
                        timeout=self.timeout,
                        temperature=self.temperature,
                        max_tokens=self.max_output_tokens,
                    ),
                )

                headers = {"Content-Type": "application/json"}
                response = requests.post(self.url, json=data, headers=headers)
                if response.status_code == 200:
                    result = response.json()
                    outputs = result["texts"]
                else:
                    print("Error:", response.status_code, response.text)
            except Exception as e:
                print("CloudModel request failed:", e)
                time.sleep(1)

        return outputs


class LangDetectModel(EvalModel):
    engine: str = ""

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        text = " ".join([x for x in inputs if isinstance(x, str)])
        try:
            return langdetect.detect(text)
        except LangDetectException as e:
            if "No features in text." not in str(e):
                print(e)
            return ""


class FastTextModel(EvalModel):
    # noinspection PyProtectedMember
    model: Optional[fasttext.FastText._FastText] = None
    engine: str = "facebook/fasttext-language-identification"

    def load(self):
        if self.model is None:
            model_path = hf_hub_download(repo_id=self.engine, filename="model.bin")
            self.model = fasttext.load_model(model_path)

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()
        texts = [
            x for i in inputs if isinstance(i, str) for x in i.split("\n") if x.strip()
        ]
        if not texts:
            return ""

        outputs, _ = self.model.predict(texts)
        labels = [o[0].replace("__label__", "") for o in outputs]
        labels = [dict(eng_Latn="en").get(x, x) for x in labels]
        assert len(labels) == len(texts)
        return max(set(labels), key=labels.count)


class LlavaModel(EvalModel):
    engine: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    model: Optional[vllm.LLM] = None

    def load(self):
        if self.model is None:
            self.model = vllm.LLM(
                model=self.engine,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 100},
                dtype="auto",
            )
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)

    @staticmethod
    def make_prompt_and_images(
        inputs: List[Union[str, Image.Image]]
    ) -> tuple[str, list[Image.Image], None]:
        # Adapted from: https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf
        text = "\n\n".join([x for x in inputs if isinstance(x, str)])
        placeholders = "".join(
            ["<image>" for x in inputs if isinstance(x, Image.Image)]
        )
        prompt = (
            f"<|im_start|>user {placeholders}\n{text}<|im_end|><|im_start|>assistant\n"
        )
        print(prompt)
        images = [x for x in inputs if isinstance(x, Image.Image)]
        stop_token_ids = None
        return prompt, images, stop_token_ids

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()
        prompt, images, stop_token_ids = self.make_prompt_and_images(inputs)
        sampling_params = vllm.SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            stop_token_ids=stop_token_ids,
        )

        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": images},
        }

        outputs = self.model.generate(inputs, sampling_params=sampling_params)
        return outputs[0].outputs[0].text


class InternSmallModel(EvalModel):
    engine: str = "OpenGVLab/InternVL2-8B"
    model: Optional[vllm.LLM] = None
    tokenizer: Optional[PreTrainedTokenizer] = None
    image_size: int = 448

    def load(self):
        if self.model is None:
            self.model = vllm.LLM(
                model=self.engine,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 100},
                dtype="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.engine, trust_remote_code=True
            )

    def make_prompt_and_images(
        self, inputs: List[Union[str, Image.Image]]
    ) -> tuple[
        str | List[int] | List[str] | List[List[int]] | BatchEncoding,
        list[Image.Image],
        list[int | List[int]],
    ]:
        # Adapted from: https://huggingface.co/OpenGVLab/InternVL2-2B
        text = "\n\n".join([x for x in inputs if isinstance(x, str)])
        placeholders = "\n".join(
            f"Image-{i}: <image>"
            for i, x in enumerate(inputs, start=1)
            if isinstance(x, Image.Image)
        )
        messages = [{"role": "user", "content": f"{placeholders}\n{text}"}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = [x for x in inputs if isinstance(x, Image.Image)]

        stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        stop_token_ids = [self.tokenizer.convert_tokens_to_ids(i) for i in stop_tokens]
        return prompt, images, stop_token_ids

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()
        prompt, images, stop_token_ids = self.make_prompt_and_images(inputs)

        sampling_params = vllm.SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            stop_token_ids=stop_token_ids,
        )

        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": images},
        }

        outputs = self.model.generate(inputs, sampling_params=sampling_params)
        return outputs[0].outputs[0].text


class PhiModel(EvalModel):
    path: str = "models/phi"
    engine: str = "microsoft/Phi-3.5-vision-instruct"
    model: Optional[vllm.LLM] = None
    image_size: int = 768

    def load(self):
        if self.model is None:
            path = self.path if os.path.exists(self.path) else self.engine
            print(dict(load_path=path))
            self.model = vllm.LLM(
                model=path,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 100},
                max_model_len=64128,
                dtype="auto",
            )

    def make_prompt_and_images(
        self, inputs: List[Union[str, Image.Image]]
    ) -> tuple[str, list[Image.Image]]:
        # Adapted from: https://huggingface.co/microsoft/Phi-3.5-vision-instruct
        text = "\n\n".join([x for x in inputs if isinstance(x, str)])
        placeholders = "\n".join(
            f"<|image_{i}|>"
            for i, x in enumerate(inputs, start=1)
            if isinstance(x, Image.Image)
        )
        prompt = f"<|user|>\n{placeholders}\n{text}<|end|>\n<|assistant|>\n"
        images = [
            resize_image(x, self.image_size)
            for x in inputs
            if isinstance(x, Image.Image)
        ]
        return prompt, images

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()
        prompt, images = self.make_prompt_and_images(inputs)
        sampling_params = vllm.SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )

        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": images},
        }

        outputs = self.model.generate(inputs, sampling_params=sampling_params)
        return outputs[0].outputs[0].text


class PixtralModel(EvalModel):
    path: str = "models/pixtral"
    engine: str = "mistralai/Pixtral-12B-2409"
    model: Optional[vllm.LLM] = None
    image_size: int = 768

    def load(self):
        if self.model is None:
            path = self.path if os.path.exists(self.path) else self.engine
            print(dict(load_path=path))
            self.model = vllm.LLM(
                model=path,
                trust_remote_code=True,
                tokenizer_mode="mistral",
                limit_mm_per_prompt={"image": 100},
                dtype="auto",
            )

    def make_messages(self, inputs: List[Union[str, Image.Image]]) -> List[dict]:
        # Adapted from: https://huggingface.co/mistralai/Pixtral-12B-2409
        content = []
        for x in inputs:
            if isinstance(x, Image.Image):
                url = f"data:image/png;base64,{convert_image_to_text(resize_image(x, self.image_size))}"
                content.append(dict(type="image_url", image_url=dict(url=url)))
            elif isinstance(x, str):
                content.append(dict(type="text", text=x))
            else:
                raise ValueError

        messages = [dict(role="user", content=content)]
        return messages

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()
        messages = self.make_messages(inputs)
        sampling_params = vllm.SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
        )

        try:
            outputs = self.model.chat(messages, sampling_params=sampling_params)
            return outputs[0].outputs[0].text
        except Exception as e:
            # OverflowError: Error in model execution: out of range integral type conversion attempted
            return f"Error: {e}"


def select_model(model_name: str, **kwargs) -> EvalModel:
    model_map = dict(
        gemini=GeminiModel,
        gemini_flash=GeminiFlashModel,
        openai=OpenAIModel,
        openai_mini=OpenAIMiniModel,
        claude=ClaudeModel,
        claude_haiku=ClaudeHaikuModel,
        reka=RekaModel,
        gemma=GemmaModel,
        idefics=IdeficsModel,
        intern=InternModel,
        onevision=OneVisionModel,
        cogvlm=CogVLMModel,
        owl=OwlModel,
        fasttext=FastTextModel,
        langdetect=LangDetectModel,
        highres_onevision=HighresOneVisionModel,
        highres_intern=HighresInternModel,
        highres_idefics=HighresIdeficsModel,
        azure=AzureModel,
        azure_mini=AzureMiniModel,
        qwen=QwenModel,
        custom_qwen=CustomQwenModel,
        llava=LlavaModel,
        phi=PhiModel,
        pixtral=PixtralModel,
        intern_small=InternSmallModel,
        text_only_azure=TextOnlyAzureModel,
        text_only_qwen=TextOnlyQwenModel,
        highres_qwen=HighresQwenModel,
    )
    model_class = model_map.get(model_name)
    if model_class is None:
        return CloudModel(engine=model_name, **kwargs)
    return model_class(**kwargs)


def test_model(
    prompt: str = "Can you extract the tables from this report?",
    image_path: str = "data/demo_image_report.png",
    image_url: str = "https://english.www.gov.cn/images/202404/20/6622f970c6d0868f1ea91c82.jpeg",
    model_name: str = "openai",
    **kwargs,
):
    model = select_model(model_name, **kwargs)
    print(locals())

    if image_path:
        image = Image.open(image_path).convert("RGB")
    else:
        image = load_image_from_url(image_url)

    inputs = [prompt, image]
    print(model.run(inputs))


def test_model_on_document(
    path: str,
    name: str,
    prompt: str = "Can you explain the figures in this document in detail?",
    **kwargs,
):
    model = select_model(name, **kwargs)
    doc = MultimodalDocument.load(path)

    inputs = [prompt]
    for page in doc.pages[:5]:
        inputs.extend([o.get_image() for o in page.get_tables_and_figures()])
        # inputs.append(page.text)

    print(model.run(inputs))


def test_run_many(
    prompt: str = "Can you extract the tables from this report?",
    image_path: str = "data/demo_image_report.png",
    image_url: str = "https://english.www.gov.cn/images/202404/20/6622f970c6d0868f1ea91c82.jpeg",
    model_name: str = "openai",
    **kwargs,
):
    model = select_model(model_name, **kwargs)
    print(locals())

    if image_path:
        image = Image.open(image_path).convert("RGB")
    else:
        image = load_image_from_url(image_url)

    inputs = [prompt, image]
    print(model.run_many(inputs, num=3))


"""
# Download locally to save time for DLC servers
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir models/qwen
huggingface-cli download mistralai/Pixtral-12B-2409 --local-dir models/pixtral
huggingface-cli download microsoft/Phi-3.5-vision-instruct --local-dir models/phi
huggingface-cli download lmms-lab/llava-onevision-qwen2-7b-ov --local-dir models/onevision
huggingface-cli download TIGER-Lab/Mantis-8B-Idefics2 --local-dir models/idefics

p modeling.py test_model --model_name gemini
p modeling.py test_model --model_name openai
p modeling.py test_model --model_name claude
p modeling.py test_model --model_name reka
p modeling.py test_model --model_name idefics
p modeling.py test_model --model_name openai_mini
p modeling.py test_model --model_name gemini_flash
p modeling.py test_model --model_name claude_haiku
p modeling.py test_model --model_name intern
p modeling.py test_model --model_name gemma
p modeling.py test_model --model_name onevision
p modeling.py test_model --model_name fasttext
p modeling.py test_model --model_name langdetect

# Single-image
python modeling.py test_model --model_name gpt-4o-2024-05-13
python modeling.py test_model --model_name claude-3-5-sonnet-20240620
python modeling.py test_model --model_name gemini-1.5-pro-001
python modeling.py test_model --model_name reka-core-20240501
python modeling.py test_model --model_name cogvlm
python modeling.py test_model --model_name owl (not very good)
python modeling.py test_model --model_name intern_small 

# Run many outputs
p modeling.py test_run_many --model_name gemini-1.5-pro-001
p modeling.py test_run_many --model_name gpt-4o-2024-05-13
p modeling.py test_run_many --model_name claude-3-5-sonnet-20240620

p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name azure
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name claude-3-5-sonnet-20240620
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name intern (good)
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name onevision (not very good)
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name idefics (good)
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name reka-core-20240501 (error for > 6 images)
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name cogvlm (multi-image not supported)
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name owl (bad)
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name qwen
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name custom_qwen
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name intern_small
python modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name llava
python modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name phi
python modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name pixtral

"""


if __name__ == "__main__":
    Fire()
