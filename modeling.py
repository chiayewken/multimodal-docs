import os
import time
from typing import Optional, List, Union

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
)

from data_loading import load_image_from_url, convert_image_to_text, MultimodalDocument

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
        content = []

        for x in inputs:
            if isinstance(x, str):
                content.append(dict(type="text", text=x))
            elif isinstance(x, Image.Image):
                content.append(
                    dict(
                        type="image_url",
                        image_url=f"data:image/png;base64,{convert_image_to_text(x)}",
                    )
                )
            else:
                raise ValueError(f"Unsupported input type: {type(x)}")

        return [dict(content=content, role="user")]

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
    engine: str = "HuggingFaceM4/idefics2-8b-chatty"
    model: Optional[Idefics2ForConditionalGeneration] = None
    processor: Optional[Idefics2Processor] = None
    device: str = "cuda"

    def load(self):
        if self.model is None:
            self.model = Idefics2ForConditionalGeneration.from_pretrained(
                self.engine, torch_dtype=torch.float16
            )
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
                do_sample=True,  # Otherwise the outputs will be very repetitive
            )
            texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
            return texts[0].split("\nAssistant:", maxsplit=1)[1].strip()


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
        else:
            raise ValueError(f"Unknown engine: {self.engine}")

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

# CloudModel API
python modeling.py test_model --model_name gpt-4o-2024-05-13
python modeling.py test_model --model_name claude-3-5-sonnet-20240620
python modeling.py test_model --model_name gemini-1.5-pro-001

# Run many outputs
p modeling.py test_run_many --model_name gemini-1.5-pro-001
p modeling.py test_run_many --model_name gpt-4o-2024-05-13
p modeling.py test_run_many --model_name claude-3-5-sonnet-20240620

p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name claude-3-5-sonnet-20240620
p modeling.py test_model_on_document data/test/NYSE_FBHS_2023.json --name intern
"""


if __name__ == "__main__":
    Fire()
