import json
import time
from typing import Optional, List

import google.generativeai as genai
from PIL import Image
from fire import Fire
from openai import OpenAI
from pydantic import BaseModel

from data_loading import MultimodalObject, MultimodalDocument, load_image_from_url


class EvalModel(BaseModel, arbitrary_types_allowed=True):
    model_path: str
    temperature: float = 0.0

    def run(self, inputs: MultimodalDocument) -> str:
        raise NotImplementedError


class GeminiModel(EvalModel):
    model_path: str = "gemini_info.json"
    timeout: int = 60
    model: Optional[genai.GenerativeModel]

    def load(self):
        if self.model is None:
            with open(self.model_path) as f:
                info = json.load(f)
                genai.configure(api_key=info["key"])
                self.model = genai.GenerativeModel(info["engine"])

    def run(self, inputs: MultimodalDocument) -> str:
        self.load()
        output = ""
        config = genai.types.GenerationConfig(
            candidate_count=1,
            temperature=self.temperature,
        )

        content = []
        for x in inputs.objects:
            if x.text:
                content.append(x.text)
            if x.image_string:
                content.append(x.get_image())

        while not output:
            try:
                response = self.model.generate_content(
                    content, generation_config=config
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


class GeminiVisionModel(GeminiModel):
    model_path = "gemini_vision_info.json"


class OpenAIModel(EvalModel):
    model_path: str = "openai_info.json"
    timeout: int = 60
    engine: str = ""
    client: Optional[OpenAI]

    def load(self):
        with open(self.model_path) as f:
            info = json.load(f)
            self.engine = info["engine"]
            self.client = OpenAI(api_key=info["key"], timeout=self.timeout)

    def make_messages(self, inputs: MultimodalDocument) -> List[dict]:
        content = []

        for x in inputs.objects:
            if x.text:
                content.append({"type": "text", "text": x.text})
            if x.image_string:
                assert "vision" in self.engine
                url = f"data:image/jpeg;base64,{x.image_string}"
                content.append({"type": "image_url", "image_url": {"url": url}})

        return [{"role": "user", "content": content}]

    def run(self, inputs: MultimodalDocument) -> str:
        self.load()
        output = ""
        error_message = "The response was filtered"

        while not output:
            try:
                response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=self.make_messages(inputs),
                    temperature=self.temperature,
                    max_tokens=512,
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


class OpenAIVisionModel(OpenAIModel):
    model_path = "openai_vision_info.json"


def select_model(model_name: str, **kwargs) -> EvalModel:
    model_map = dict(
        gemini=GeminiModel,
        gemini_vision=GeminiVisionModel,
        openai=OpenAIModel,
        openai_vision=OpenAIVisionModel,
    )
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"{model_name}. Choose from {list(model_map.keys())}")
    return model_class(**kwargs)


def test_model(
    prompt: str = "Can you describe this image in detail?",
    image_path: str = "",
    image_url: str = "https://github.com/UKPLab/sentence-transformers/raw/master/examples/applications/image-search/two_dogs_in_snow.jpg",
    model_name: str = "gemini_vision",
    **kwargs,
):
    model = select_model(model_name, **kwargs)
    print(locals())

    if image_path:
        image = Image.open(image_path)
    else:
        image = load_image_from_url(image_url)

    inputs = MultimodalDocument(
        objects=[MultimodalObject(text=prompt), MultimodalObject.from_image(image)]
    )
    print(model.run(inputs))


"""
p modeling.py test_model --model_name gemini_vision
p modeling.py test_model --model_name openai
p modeling.py test_model --model_name openai_vision
"""


if __name__ == "__main__":
    Fire()
