from typing import Optional

from fire import Fire
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

    def load(self):
        if self.model is None:
            self.model = SentenceTransformer("clip-ViT-L-14")

    def run(
        self, query: MultimodalObject, doc: MultimodalDocument
    ) -> MultimodalDocument:
        self.load()
        doc = doc.copy(deep=True)
        texts = [x.text for x in doc.objects]
        query = self.model.encode(query.text)
        data = self.model.encode(texts)
        scores = cos_sim(query, data).squeeze().tolist()
        assert len(scores) == len(doc.objects)

        for i, o in enumerate(doc.objects):
            o.score = scores[i]
        return doc


def select_retriever(name: str, **kwargs) -> MultimodalRetriever:
    if name == "clip_text":
        return ClipTextRetriever(**kwargs)
    raise KeyError(name)


def test_retriever(
    name: str = "clip_text",
    query: str = "How many dogs are there?",
    image_url: str = "https://github.com/UKPLab/sentence-transformers/raw/master/examples/applications/image-search/two_dogs_in_snow.jpg",
    **kwargs
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
