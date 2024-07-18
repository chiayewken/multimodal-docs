import hashlib
from typing import Optional, List, Dict

import numpy as np
import torch
from PIL import Image
from fire import Fire
from nltk import sent_tokenize
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    PaliGemmaPreTrainedModel,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    SiglipModel,
    SiglipProcessor,
)

from data_loading import MultimodalObject, MultimodalDocument
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
        assert len(scores) == len(doc.objects)
        for i, o in enumerate(doc.objects):
            o.score = scores[i]

        doc.objects = sorted(doc.objects, key=lambda x: x.score)[::-1][: self.top_k]
        doc.objects = sorted(doc.objects, key=lambda x: x.page)
        return doc


class SiglipRetriever(MultimodalRetriever):
    path: str = "google/siglip-so400m-patch14-384"
    model: Optional[SiglipModel] = None
    processor: Optional[SiglipProcessor] = None
    device: str = "cuda"
    cache: Dict[str, BM25Okapi] = {}

    def load(self):
        if self.model is None:
            self.model = SiglipModel.from_pretrained(self.path)
            self.model = self.model.to(self.device).eval()
            self.processor = SiglipProcessor.from_pretrained(self.path)

    def run(
        self, query: MultimodalObject, doc: MultimodalDocument
    ) -> MultimodalDocument:
        doc = doc.copy(deep=True)
        self.load()
        images = [x.get_image().convert("RGB") for x in doc.objects if x.image_string]
        indices = [i for i, x in enumerate(doc.objects) if x.image_string]
        inputs = self.processor(
            text=[query.text],
            images=images,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

        with torch.no_grad():
            # noinspection PyTypeChecker
            outputs = self.model(**inputs.to(self.device))
        logits_per_image = outputs.logits_per_image
        scores = torch.sigmoid(logits_per_image).squeeze()  # probabilities
        assert scores.ndim == 1
        assert len(scores) == len(indices)
        for i, s in zip(indices, scores):
            doc.objects[i].score = float(s)

        doc.objects = sorted(doc.objects, key=lambda x: x.score)[::-1][: self.top_k]
        doc.objects = sorted(doc.objects, key=lambda x: x.page)
        return doc


class ColPali(PaliGemmaPreTrainedModel):
    def __init__(self, config):
        super(ColPali, self).__init__(config=config)
        self.model: PaliGemmaForConditionalGeneration = (
            PaliGemmaForConditionalGeneration(config)
        )
        self.dim = 128
        self.custom_text_proj = nn.Linear(
            self.model.config.text_config.hidden_size, self.dim
        )
        self.main_input_name = "doc_input_ids"

    def forward(self, *args, **kwargs):
        """
        Forward pass through Llama and the linear layer for dimensionality reduction

        Args:
        - input_ids (torch.LongTensor): The input tokens tensor.
        - attention_mask (torch.LongTensor): The attention mask tensor.

        Returns:
        - torch.Tensor: Embeddings of shape (batch_size, num_tokens, dim)
        """
        outputs = self.model(*args, output_hidden_states=True, **kwargs)
        last_hidden_states = outputs.hidden_states[
            -1
        ]  # (batch_size, sequence_length, hidden_size)
        proj = self.custom_text_proj(last_hidden_states)
        # normalize l2 norm
        proj = proj / proj.norm(dim=-1, keepdim=True)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)
        return proj


class CustomEvaluator:
    def evaluate(self, qs, ps):
        scores = self.evaluate_colbert(qs, ps)
        assert scores.shape[0] == len(qs)
        scores = scores.to(torch.float32).cpu().numpy()
        return scores

    @staticmethod
    def evaluate_colbert(qs, ps, batch_size=128) -> torch.Tensor:
        scores = []
        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = torch.nn.utils.rnn.pad_sequence(
                qs[i : i + batch_size], batch_first=True, padding_value=0
            ).to("cuda")
            for j in range(0, len(ps), batch_size):
                ps_batch = torch.nn.utils.rnn.pad_sequence(
                    ps[j : j + batch_size], batch_first=True, padding_value=0
                ).to("cuda")
                scores_batch.append(
                    torch.einsum("bnd,csd->bcns", qs_batch, ps_batch)
                    .max(dim=3)[0]
                    .sum(dim=2)
                )
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores.append(scores_batch)
        scores = torch.cat(scores, dim=0)
        return scores


class ColpaliRetriever(MultimodalRetriever):
    model: Optional[ColPali] = None
    processor: Optional[PaliGemmaProcessor] = None
    device: str = "cuda"
    cache: Dict[str, list] = {}

    def load(self):
        if self.model is None:
            self.model = ColPali.from_pretrained(
                "google/paligemma-3b-mix-448", torch_dtype=torch.bfloat16
            )
            self.model = self.model.to(self.device).eval()
            self.model.load_adapter("vidore/colpali")
            self.processor = PaliGemmaProcessor.from_pretrained("vidore/colpali")

    @staticmethod
    def process_images(processor, images, max_length: int = 50):
        texts_doc = ["Describe the image."] * len(images)
        images = [image.convert("RGB") for image in images]

        batch_doc = processor(
            text=texts_doc,
            images=images,
            return_tensors="pt",
            padding="longest",
            max_length=max_length + processor.image_seq_length,
        )
        return batch_doc

    @staticmethod
    def process_queries(processor, queries, mock_image, max_length: int = 50):
        texts_query = []
        for query in queries:
            query = f"Question: {query}<unused0><unused0><unused0><unused0><unused0>"
            texts_query.append(query)

        batch_query = processor(
            images=[mock_image.convert("RGB")] * len(texts_query),
            # NOTE: the image is not used in batch_query but it is required for calling the processor
            text=texts_query,
            return_tensors="pt",
            padding="longest",
            max_length=max_length + processor.image_seq_length,
        )
        del batch_query["pixel_values"]

        batch_query["input_ids"] = batch_query["input_ids"][
            ..., processor.image_seq_length :
        ]
        batch_query["attention_mask"] = batch_query["attention_mask"][
            ..., processor.image_seq_length :
        ]
        return batch_query

    def load_document_images(self, images: List[Image.Image]):
        hash_id = "".join([hashlib.md5(i.tobytes()).hexdigest() for i in images])
        if hash_id in self.cache:
            return self.cache[hash_id]

        # noinspection PyTypeChecker
        dataloader = DataLoader(
            images,
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: self.process_images(self.processor, x),
        )

        ds = []
        for batch_doc in dataloader:
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

        self.cache[hash_id] = ds
        return ds

    def run(
        self, query: MultimodalObject, doc: MultimodalDocument
    ) -> MultimodalDocument:
        doc = doc.copy(deep=True)
        self.load()
        images = [x.get_image() for x in doc.objects if x.image_string]
        indices = [i for i, x in enumerate(doc.objects) if x.image_string]
        ds = self.load_document_images(images)

        # run inference - queries
        # noinspection PyTypeChecker
        dataloader = DataLoader(
            [query.text],
            batch_size=4,
            shuffle=False,
            collate_fn=lambda x: self.process_queries(
                self.processor, x, Image.new("RGB", (448, 448), (255, 255, 255))
            ),
        )

        qs = []
        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {
                    k: v.to(self.model.device) for k, v in batch_query.items()
                }
                embeddings_query = self.model(**batch_query)
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

        retriever_evaluator = CustomEvaluator()
        scores = retriever_evaluator.evaluate(qs, ds).squeeze()
        assert scores.ndim == 1
        assert len(scores) == len(indices)
        for i, s in zip(indices, scores):
            doc.objects[i].score = float(s)

        doc.objects = sorted(doc.objects, key=lambda x: x.score)[::-1][: self.top_k]
        doc.objects = sorted(doc.objects, key=lambda x: x.page)
        return doc


def select_retriever(name: str, **kwargs) -> MultimodalRetriever:
    if name == "clip_text":
        return ClipTextRetriever(**kwargs)
    elif name == "page":
        return PageRetriever(**kwargs)
    elif name == "bm25":
        return BM25PageRetriever(**kwargs)
    elif name == "colpali":
        return ColpaliRetriever(**kwargs)
    elif name == "siglip":
        return SiglipRetriever(**kwargs)
    raise KeyError(name)


def test_retriever(name: str = "clip_text", **kwargs):
    generator = OpenAIModel()
    retriever = select_retriever(name, top_k=2, **kwargs)

    doc = MultimodalDocument(
        objects=[
            MultimodalObject(text="The dogs are playing in the snow"),
            MultimodalObject(text="A cat on the table"),
            MultimodalObject(text="A picture of an event"),
        ]
    )

    for path in [
        "data/demo_image_dogs.png",
        "data/demo_image_ceremony.jpeg",
        "data/demo_image_report.png",
    ]:
        doc.objects.append(MultimodalObject.from_image(Image.open(path), text=path))

    for query in [
        "How many people are there at the ceremony?",
        "What are the animals doing?",
        "Which year had more equity?",
    ]:
        context = retriever.run(MultimodalObject(text=query), doc)
        for o in context.objects:
            print(o.dict(exclude={"image_string"}))
        context.objects.insert(0, MultimodalObject(text=query))

        inputs = [x.get_image() or x.text for x in context.objects]
        outputs = generator.run(inputs)
        print(dict(query=query))
        print(dict(outputs=outputs))
        print()


if __name__ == "__main__":
    Fire()
