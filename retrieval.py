from typing import Optional

import numpy as np
from fire import Fire
from nltk import sent_tokenize
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from data_loading import MultimodalObject, MultimodalDocument, load_image_from_url
from modeling import GeminiVisionModel


class MultimodalRetriever(BaseModel, arbitrary_types_allowed=True):
    def run(
        self, query: MultimodalObject, doc: MultimodalDocument
    ) -> MultimodalDocument:
        raise NotImplementedError


class ClipTextRetriever(MultimodalRetriever):
    model_path: str = "clip-ViT-L-14"
    model: Optional[SentenceTransformer] = None
    max_length: int = 77

    def load(self):
        if self.model is None:
            self.model = SentenceTransformer("clip-ViT-L-14")

    def check_length(self, text) -> int:
        return self.model.tokenize([text]).input_ids.shape[1]

    def truncate_text(self, text: str, sep: str = " ") -> str:
        if self.check_length(text) <= self.max_length:
            return text

        words = text.split(sep)
        while self.check_length(sep.join(words)) > self.max_length:
            words = words[:-1]

        new = sep.join(words)
        print(dict(original=self.check_length(text), truncated=self.check_length(new)))
        return new

    def run(
        self, query: MultimodalObject, doc: MultimodalDocument
    ) -> MultimodalDocument:
        self.load()
        doc = doc.copy(deep=True)
        query_embeds = self.model.encode(query.text)

        # Split long texts to avoid CLIP max context length error (77 tokens)
        groups = {}
        for i, x in enumerate(doc.objects):
            if x.image_string:
                x.text = self.truncate_text(x.text)
                groups[i] = [x]
            else:
                assert x.text
                for text in sent_tokenize(x.text):
                    o = MultimodalObject(page=x.page, text=self.truncate_text(text))
                    groups.setdefault(i, []).append(o)

        # If an object text is split into multiple parts, then we select the highest scoring part
        for i, objects in groups.items():
            embeds = np.stack([self.model.encode(x.text) for x in objects])
            # noinspection PyTypeChecker
            array = cos_sim(query_embeds, embeds)
            assert array.ndim == 2
            scores = array.tolist()[0]
            j = scores.index(max(scores))
            doc.objects[i].score = max(scores)
            doc.objects[i].snippet = objects[j].text

        return doc


def select_retriever(name: str, **kwargs) -> MultimodalRetriever:
    if name == "clip_text":
        return ClipTextRetriever(**kwargs)
    raise KeyError(name)


def test_retriever(
    name: str = "clip_text",
    query: str = "How many dogs are there?",
    image_url: str = "https://github.com/UKPLab/sentence-transformers/raw/master/examples/applications/image-search/two_dogs_in_snow.jpg",
    **kwargs,
):
    generator = GeminiVisionModel()
    retriever = select_retriever(name, **kwargs)
    image = load_image_from_url(image_url)

    doc = MultimodalDocument(
        objects=[
            MultimodalObject(text="The dogs are playing in the snow"),
            MultimodalObject(text="A cat on the table"),
            MultimodalObject(text="A picture of London at night"),
            MultimodalObject.from_image(image),
        ]
    )

    context = retriever.run(MultimodalObject(text=query), doc).get_top_objects(k=2)
    context.objects.insert(0, MultimodalObject(text=query))
    context.print()
    print(generator.run(context))


if __name__ == "__main__":
    Fire()
