import os
import time
from typing import Optional, List, Union

import google.generativeai as genai
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

from data_loading import load_image_from_url, convert_image_to_text


def get_environment_key(path: str, name: str) -> str:
    assert os.path.exists(path), f"Path {path} does not exist"
    load_dotenv(path)
    return os.environ[name]


class EvalModel(BaseModel, arbitrary_types_allowed=True):
    engine: str
    timeout: int = 60
    temperature: float = 0.0
    max_output_tokens: int = 512

    def run(self, inputs: List[Union[str, Image.Image]]) -> str:
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
            self.processor = Idefics2Processor.from_pretrained(
                self.engine, size={"longest_edge": 700, "shortest_edge": 378}
            )
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)

    def process_inputs(self, inputs: List[Union[str, Image.Image]]):
        self.load()
        content = []
        for x in inputs:
            if isinstance(x, str):
                content.append(dict(type="text", text=x))
            elif isinstance(x, Image.Image):
                content.append(dict(type="image"))
            else:
                raise ValueError(f"Unsupported input type: {type(x)}")

        messages = [dict(role="user", content=content)]
        images = [x for x in inputs if isinstance(x, Image.Image)]
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
            return texts[0].split(" \nAssistant: ", maxsplit=1)[1]


def select_model(model_name: str, **kwargs) -> EvalModel:
    model_map = dict(
        gemini=GeminiModel,
        openai=OpenAIModel,
        claude=ClaudeModel,
        reka=RekaModel,
        gemma=GemmaModel,
        idefics=IdeficsModel,
    )
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"{model_name}. Choose from {list(model_map.keys())}")
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
        image = Image.open(image_path)
    else:
        image = load_image_from_url(image_url)

    inputs = [prompt, image]
    print(model.run(inputs))


"""
p modeling.py test_model --model_name gemini
p modeling.py test_model --model_name openai
p modeling.py test_model --model_name claude
p modeling.py test_model --model_name reka
p modeling.py test_model --model_name idefics
"""


if __name__ == "__main__":
    Fire()
