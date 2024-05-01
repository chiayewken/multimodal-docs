from typing import Optional, List, Dict

import numpy as np
from fire import Fire
from nltk import sent_tokenize
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from data_loading import MultimodalObject, MultimodalDocument, load_image_from_url
from modeling import OpenAIModel


class MultimodalRetriever(BaseModel, arbitrary_types_allowed=True):
    top_k: int

    def run(
        self, query: MultimodalObject, doc: MultimodalDocument
    ) -> MultimodalDocument:
        raise NotImplementedError

    def get_top_objects(self, doc: MultimodalDocument):
        # Get top-k in terms of score but maintain the original order
        doc = doc.copy(deep=True)
        objects = sorted(doc.objects, key=lambda x: x.score, reverse=True)
        threshold = objects[: self.top_k][-1].score
        return MultimodalDocument(
            objects=[x for x in doc.objects if x.score >= threshold]
        )


class ClipTextRetriever(MultimodalRetriever):
    model_path: str = "clip-ViT-L-14"
    model: Optional[SentenceTransformer] = None
    max_length: int = 77

    def load(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_path)

    def check_length(self, text) -> int:
        return self.model.tokenize([text])["input_ids"].shape[1]

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


class PageRetriever(ClipTextRetriever):
    model_path: str = "sentence-transformers/all-mpnet-base-v2"
    embed_cache: Dict[str, np.ndarray] = {}

    def text_split(self, text: str) -> List[str]:
        parts = []
        for sent in sent_tokenize(text):
            for chunk in sent.split("\n"):
                if chunk.strip():
                    parts.append(self.truncate_text(chunk.strip()))
        return parts

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        self.load()
        key = "".join(texts)
        if key not in self.embed_cache:
            self.embed_cache[key] = self.model.encode(texts)
        return self.embed_cache[key]

    def run(
        self, query: MultimodalObject, doc: MultimodalDocument
    ) -> MultimodalDocument:
        self.load()
        doc = doc.copy(deep=True)
        query_embeds = self.model.encode(query.text)

        # Split long texts to avoid CLIP max context length error (77 tokens)
        groups = {}
        for i, x in enumerate(doc.objects):
            if not x.image_string:
                for text in self.text_split(x.text):
                    o = MultimodalObject(page=x.page, text=text)
                    groups.setdefault(i, []).append(o)

        # Assign the relevance of each object as the top-scoring text chunk
        for i, objects in groups.items():
            embeds = self.embed_texts([x.text for x in objects])
            # noinspection PyTypeChecker
            array = cos_sim(query_embeds, embeds)
            assert array.ndim == 2
            scores = array.tolist()[0]
            j = scores.index(max(scores))
            doc.objects[i].score = max(scores)
            doc.objects[i].snippet = objects[j].text
            for j, o in enumerate(objects):
                o.score = scores[j]

        # Assign the relevance of each page as the top-scoring object
        page_scores = {}
        for o in doc.objects:
            assert o.page != 0
            page_scores.setdefault(o.page, 0)
            page_scores[o.page] = max(page_scores[o.page], o.score)

        top_pages = sorted(page_scores, key=lambda p: page_scores[p])[-self.top_k :]
        doc.objects = [
            x
            for x in doc.objects
            if (x.page in top_pages or (x.page - 1 in top_pages and x.image_string))
        ]
        print(dict(query=query.text))
        for objects in groups.values():
            for o in objects:
                if o.score == page_scores[o.page] and o.page in top_pages:
                    print(dict(page=o.page, top_text=o.text, score=o.score))

        return doc


class BM25PageRetriever(MultimodalRetriever):
    cache: Dict[str, BM25Okapi] = {}

    def load_ranker(self, doc: MultimodalDocument) -> BM25Okapi:
        corpus = [o.text.split() for o in doc.objects]
        key = str(corpus)
        if key not in self.cache:
            self.cache[key] = BM25Okapi(corpus)

        return self.cache[key]

    def run(
        self, query: MultimodalObject, doc: MultimodalDocument
    ) -> MultimodalDocument:
        doc = doc.copy(deep=True)
        ranker = self.load_ranker(doc)
        scores = ranker.get_scores(query.text.split())

        # Assign the relevance of each page as the top-scoring object
        page_scores = {}
        assert len(scores) == len(doc.objects)
        for i, o in enumerate(doc.objects):
            assert o.page != 0
            o.score = scores[i]
            page_scores.setdefault(o.page, 0)
            page_scores[o.page] = max(page_scores[o.page], o.score)

        top_pages = sorted(page_scores, key=lambda p: page_scores[p])[-self.top_k :]
        doc.objects = [
            x
            for x in doc.objects
            if (x.page in top_pages or (x.page - 1 in top_pages and x.image_string))
        ]
        print(dict(query=query.text))
        for o in doc.objects:
            if o.score == page_scores[o.page] and o.page in top_pages:
                print(dict(page=o.page, top_text=o.text, score=o.score))

        return doc


def select_retriever(name: str, **kwargs) -> MultimodalRetriever:
    if name == "clip_text":
        return ClipTextRetriever(**kwargs)
    elif name == "page":
        return PageRetriever(**kwargs)
    elif name == "bm25_page":
        return BM25PageRetriever(**kwargs)
    raise KeyError(name)


def test_retriever(
    name: str = "clip_text",
    query: str = "How many people are there?",
    image_url: str = "https://english.www.gov.cn/images/202404/20/6622f970c6d0868f1ea91c82.jpeg",
    **kwargs,
):
    generator = OpenAIModel()
    retriever = select_retriever(name, top_k=2, **kwargs)
    image = load_image_from_url(image_url)

    doc = MultimodalDocument(
        objects=[
            MultimodalObject(text="The dogs are playing in the snow"),
            MultimodalObject(text="A cat on the table"),
            MultimodalObject(text="A picture of an event"),
            MultimodalObject.from_image(image),
        ]
    )

    context = retriever.run(MultimodalObject(text=query), doc)
    context.objects.insert(0, MultimodalObject(text=query))
    context.print()
    print(generator.run(context))


if __name__ == "__main__":
    Fire()
