import copy
import os
import time
import warnings
from typing import Optional, List, Union, Any

import google.generativeai as genai
import requests
import torch
from PIL import Image
from anthropic import Anthropic
from dotenv import load_dotenv
from fire import Fire
from openai import OpenAI
from pydantic import BaseModel
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
    client: Optional[genai.GenerativeModel]

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

    def load(self):
        if self.client is None:
            backend_config = lmdeploy.TurbomindEngineConfig(model_format="awq")
            self.client = lmdeploy.pipeline(self.engine, backend_config=backend_config)

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()
        config = lmdeploy.GenerationConfig(max_new_tokens=self.max_output_tokens)
        text = "\n\n".join([x for x in inputs if isinstance(x, str)])
        images = [resize_image(x, 448) for x in inputs if isinstance(x, Image.Image)]
        response = self.client((text, images), generation_config=config)
        return response.text


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
    engine: str = "lmms-lab/llava-onevision-qwen2-7b-ov"
    tokenizer: Optional[PreTrainedTokenizer] = None
    model: Optional[PreTrainedModel] = None
    processor: Optional[ProcessorMixin] = None

    def load(self):
        if self.model is None:
            (
                self.tokenizer,
                self.model,
                self.processor,
                _,
            ) = load_pretrained_model(self.engine, None, "llava_qwen")
            self.model.eval()

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
        self.load()
        warnings.filterwarnings("ignore")
        images = [resize_image(x, 384) for x in inputs if isinstance(x, Image.Image)]
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


class GemmaModel(EvalModel):
    engine: str = "google/paligemma-3b-mix-448"
    model: Optional[PaliGemmaForConditionalGeneration] = None
    processor: Optional[PaliGemmaProcessor] = None
    device: str = "cuda"

    def load(self):
        if self.model is None:
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
    engine: str = "TIGER-Lab/Mantis-8B-Idefics2"  # Optimized for long interleaved cases
    model: Optional[Idefics2ForConditionalGeneration] = None
    processor: Optional[Idefics2Processor] = None
    device: str = "cuda"

    def load(self):
        if self.model is None:
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
        images = [resize_image(x, 384) for x in inputs if isinstance(x, Image.Image)]
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        return self.processor(text=prompt, images=images, return_tensors="pt").to(
            self.device
        )

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


class CloudModel(EvalModel):
    # Queries the server on google cloud for completions, which should work in any country
    url: str = "http://35.208.161.228:8000/completions"

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
    prompt: str = "Can you explain the figures in this document?",
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

# Single-image
python modeling.py test_model --model_name gpt-4o-2024-05-13
python modeling.py test_model --model_name claude-3-5-sonnet-20240620
python modeling.py test_model --model_name gemini-1.5-pro-001
python modeling.py test_model --model_name reka-core-20240501
python modeling.py test_model --model_name cogvlm
python modeling.py test_model --model_name owl (not very good)

# Run many outputs
p modeling.py test_run_many --model_name gemini-1.5-pro-001
p modeling.py test_run_many --model_name gpt-4o-2024-05-13
p modeling.py test_run_many --model_name claude-3-5-sonnet-20240620

p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name claude-3-5-sonnet-20240620
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name intern (good)
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name onevision (not very good)
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name idefics (good)
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name reka-core-20240501 (error for > 6 images)
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name cogvlm (multi-image not supported)
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name owl (bad)

"""


if __name__ == "__main__":
    Fire()
