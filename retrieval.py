import hashlib
from FlagEmbedding import BGEM3FlagModel

from typing import Optional, List, Dict

import torch
from PIL import Image
from fire import Fire
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers.util import cos_sim
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    PaliGemmaPreTrainedModel,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    AutoModel,
    PreTrainedModel,
)

from data_loading import MultimodalDocument


class MultimodalRetriever(BaseModel, arbitrary_types_allowed=True):
    def run(self, query: str, doc: MultimodalDocument) -> MultimodalDocument:
        raise NotImplementedError

    @staticmethod
    def get_top_pages(doc: MultimodalDocument, k: int) -> List[int]:
        # Get top-k in terms of score but maintain the original order
        doc = doc.copy(deep=True)
        pages = sorted(doc.pages, key=lambda x: x.score, reverse=True)
        threshold = pages[:k][-1].score
        return [p.number for p in doc.pages if p.score >= threshold]


class ClipRetriever(MultimodalRetriever):
    path: str = "jinaai/jina-clip-v1"  # Document-optimized version of CLIP
    client: Optional[PreTrainedModel] = None

    def load(self):
        if self.client is None:
            self.client = AutoModel.from_pretrained(self.path, trust_remote_code=True)
            self.client = self.client.cuda()

    def run(self, query: str, doc: MultimodalDocument) -> MultimodalDocument:
        self.load()
        doc = doc.copy(deep=True)
        query_embeds = self.client.encode_text([query])

        for page in doc.pages:
            text_embeds = self.client.encode_text([page.text])
            page.score = cos_sim(query_embeds, text_embeds).item()
            objects = page.get_tables_and_figures()

            if objects:
                image_embeds = self.client.encode_image(
                    [x.get_image() for x in objects]
                )
                image_scores = cos_sim(query_embeds, image_embeds)
                for i, o in enumerate(objects):
                    o.score = image_scores[:, i].item()
                    page.score = max(page.score, o.score)

        return doc


class BM25PageRetriever(MultimodalRetriever):
    cache: Dict[str, BM25Okapi] = {}

    def load_ranker(self, doc: MultimodalDocument) -> BM25Okapi:
        corpus = [o.text.split() for o in doc.pages]
        key = str(corpus)
        if key not in self.cache:
            self.cache[key] = BM25Okapi(corpus)

        return self.cache[key]

    def run(self, query: str, doc: MultimodalDocument) -> MultimodalDocument:
        doc = doc.copy(deep=True)
        ranker = self.load_ranker(doc)
        scores = ranker.get_scores(query.split())
        assert len(scores) == len(doc.pages)
        for i, page in enumerate(doc.pages):
            page.score = scores[i]

        return doc


class BGEM3Retriever(MultimodalRetriever):
    cache: Dict[str, dict] = {}
    client: Optional[BGEM3FlagModel] = None

    def load(self):
        if self.client is None:
            self.client = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    def embed_texts(self, texts: List[str]) -> dict:
        self.load()
        return self.client.encode(
            texts, return_dense=True, return_sparse=True, return_colbert_vecs=True
        )

    def embed_document(self, doc: MultimodalDocument) -> dict:
        texts = [page.text for page in doc.pages]
        key = str(texts)
        if key not in self.cache:
            self.cache[key] = self.embed_texts(texts)
        return self.cache[key]

    def run(self, query: str, doc: MultimodalDocument) -> MultimodalDocument:
        self.load()
        doc = doc.copy(deep=True)
        doc_embeds = self.embed_document(doc)["colbert_vecs"]
        query_embed = self.embed_texts([query])["colbert_vecs"][0]

        assert len(doc_embeds) == len(doc.pages)
        for i, page in enumerate(doc.pages):
            page.score = self.client.colbert_score(query_embed, doc_embeds[i]).item()

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

    def run(self, query: str, doc: MultimodalDocument) -> MultimodalDocument:
        doc = doc.copy(deep=True)
        self.load()
        images = [page.get_full_image() for page in doc.pages]
        ds = self.load_document_images(images)

        # run inference - queries
        # noinspection PyTypeChecker
        dataloader = DataLoader(
            [query],
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
        assert len(scores) == len(doc.pages)
        for i, page in enumerate(doc.pages):
            page.score = float(scores[i])

        return doc


class HybridRetriever(MultimodalRetriever):
    # Use Reciprocal Rank Fusion (RRF) scores to combine multiple retrievers
    models: List[MultimodalRetriever] = [
        BGEM3Retriever(),
        ColpaliRetriever(),
        BM25PageRetriever(),
    ]
    k: int = 60  # Hyperparameter

    def run(self, query: str, doc: MultimodalDocument) -> MultimodalDocument:
        doc = doc.copy(deep=True)
        results = [model.run(query, doc) for model in self.models]

        # Calculate RRF scores
        scores = dict()
        for res in results:
            sorted_pages = sorted(res.pages, key=lambda p: p.score, reverse=True)
            for rank, page in enumerate(sorted_pages, start=1):
                scores.setdefault(page.number, 0)
                scores[page.number] += 1 / (self.k + rank)

        # Update scores
        for page in doc.pages:
            page.score = scores[page.number]
        return doc


def select_retriever(name: str, **kwargs) -> MultimodalRetriever:
    if name == "clip":
        return ClipRetriever(**kwargs)
    elif name == "bm25":
        return BM25PageRetriever(**kwargs)
    elif name == "colpali":
        return ColpaliRetriever(**kwargs)
    elif name == "bge":
        return BGEM3Retriever(**kwargs)
    elif name == "hybrid":
        return HybridRetriever(**kwargs)
    raise KeyError(name)


def test_retriever(path: str = "data/test/NYSE_FBHS_2023.json", name: str = "clip"):
    retriever = select_retriever(name)
    doc = MultimodalDocument.load(path)
    doc.pages = doc.pages[:5]

    for query in [
        "What is the market capitalization?",
        "What color suit is the CEO wearing?",
        "What are all the brands under the company?",
    ]:
        output = retriever.run(query, doc)
        for i in retriever.get_top_pages(output, k=1):
            print(dict(query=query))
            print(dict(scores=[p.score for p in output.pages]))
            print(output.get_page(i).dict(exclude={"image_string", "objects"}))
        print("#" * 80)


if __name__ == "__main__":
    Fire()
