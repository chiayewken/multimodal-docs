import base64
import io
import json
from pathlib import Path
from typing import List, Optional

import requests
from PIL import Image
from fire import Fire
from pydantic import BaseModel


def convert_image_to_text(image: Image) -> str:
    # This is also how OpenAI encodes images: https://platform.openai.com/docs/guides/vision
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        data = output.getvalue()
    return base64.b64encode(data).decode("utf-8")


def convert_text_to_image(text: str) -> Image:
    data = base64.b64decode(text.encode("utf-8"))
    return Image.open(io.BytesIO(data))


def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    raise ValueError(f"Failed to retrieve image. Status code: {response.status_code}")


class MultimodalObject(BaseModel):
    page: int = 0
    text: str = ""
    image_string: str = ""
    snippet: str = ""
    score: float = 0.0

    def get_image(self) -> Optional[Image.Image]:
        if self.image_string:
            return convert_text_to_image(self.image_string)

    @classmethod
    def from_image(cls, image: Image.Image, **kwargs):
        return cls(image_string=convert_image_to_text(image), **kwargs)


class MultimodalDocument(BaseModel):
    objects: List[MultimodalObject]

    def get_top_objects(self, k: int):
        # Get top-k in terms of score but maintain the intra-document ordering
        doc = self.copy(deep=True)
        objects = sorted(doc.objects, key=lambda x: x.score, reverse=True)
        threshold = objects[:k][-1].score
        return MultimodalDocument(
            objects=[x for x in doc.objects if x.score >= threshold]
        )

    def print(self):
        for x in self.objects:
            x = x.copy(deep=True)
            if len(x.text) > 80:
                x.text = x.text[:80] + "..."
            if x.image_string:
                x.image_string = x.image_string[:20] + "..."
            print(x.json(indent=2))


class MultimodalSample(BaseModel):
    question: str
    answer: str
    doc: MultimodalDocument
    prompt: MultimodalDocument = MultimodalDocument(objects=[])
    raw_output: str = ""
    pred: str = ""

    def print(self):
        self.doc.print()
        self.prompt.print()
        print(self.json(indent=2, exclude={"prompt", "doc"}))


class MultimodalData(BaseModel):
    samples: List[MultimodalSample]

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for s in self.samples:
                print(s.json(), file=f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            samples = [MultimodalSample(**json.loads(line)) for line in f]
        return cls(samples=samples)


if __name__ == "__main__":
    Fire()
