import hashlib
import json
import random
import subprocess
import sys
from collections import OrderedDict as CollectionsOrderedDict
from pathlib import Path
from typing import Optional, List, Dict, OrderedDict

import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from PIL import Image
from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import (
    process_images,
    process_queries,
)
from fire import Fire
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers.util import cos_sim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    PaliGemmaProcessor,
    AutoModel,
    PreTrainedModel,
    AutoProcessor,
)

from data_loading import MultimodalDocument, MultimodalData, MultimodalPage


def run_shell_command(command: List[str]) -> None:
    try:
        process = subprocess.Popen(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())
                sys.stdout.flush()

        rc = process.poll()
        if rc != 0:
            print(f"Command failed with return code {rc}")
            error_output = process.stderr.read()
            print("Error output:", error_output)
        else:
            print("Command executed successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


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

    def finetune(self, data: MultimodalData, output_dir: str):
        raise NotImplementedError


class ClipRetriever(MultimodalRetriever):
    path: str = "jinaai/jina-clip-v1"  # Document-optimized version of CLIP
    client: Optional[PreTrainedModel] = None
    cache: Dict[str, torch.Tensor] = {}

    def load(self):
        if self.client is None:
            self.client = AutoModel.from_pretrained(self.path, trust_remote_code=True)
            self.client = self.client.cuda()

    def encode_page(self, page: MultimodalPage) -> torch.Tensor:
        key = page.json()
        if key not in self.cache:
            embeds = self.client.encode_text([page.text])
            objects = page.get_tables_and_figures()

            if objects:
                image_embeds = self.client.encode_image(
                    [x.get_image() for x in objects]
                )
                embeds = np.concatenate([embeds, image_embeds], axis=0)
            self.cache[key] = embeds
        return self.cache[key]

    def run(self, query: str, doc: MultimodalDocument) -> MultimodalDocument:
        self.load()
        doc = doc.copy(deep=True)
        query_embeds = self.client.encode_text([query])

        for page in doc.pages:
            page_embeds = self.encode_page(page)
            scores = cos_sim(query_embeds, page_embeds)
            page.score = scores.max().item()

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
    path: str = "BAAI/bge-m3"
    cache: OrderedDict[str, dict] = CollectionsOrderedDict()
    client: Optional[BGEM3FlagModel] = None

    def load(self):
        if self.client is None:
            self.client = BGEM3FlagModel(self.path, use_fp16=True)

    def embed_texts(self, texts: List[str]) -> dict:
        self.load()
        return self.client.encode(
            texts, return_dense=True, return_sparse=True, return_colbert_vecs=True
        )

    def embed_document(self, doc: MultimodalDocument) -> dict:
        texts = [page.text for page in doc.pages]
        key = str(texts)
        if len(self.cache) > 100:
            self.cache.popitem(last=False)
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

    def finetune(self, data: MultimodalData, output_dir: str):
        path = Path(output_dir, "data.jsonl")
        path.parent.mkdir(exist_ok=True, parents=True)
        random.seed(0)
        doc_map = {}

        with open(path, "w") as f:
            for sample in tqdm(data.samples, desc=str(path)):
                if sample.source not in doc_map:
                    doc_map[sample.source] = MultimodalDocument.load(sample.source)
                doc = doc_map[sample.source]
                page_map = {p.number: p.text for p in doc.pages}
                pool = set(j for i in sample.evidence_pages for j in [i - 1, i, i + 1])
                info = dict(
                    query=sample.question,
                    pos=[page_map[i] for i in sample.evidence_pages],
                    neg=random.sample(set(p.number for p in doc.pages) - pool, k=10),
                )
                print(json.dumps(info), file=f)

        command = [
            "torchrun",
            "--nproc_per_node",
            "1",
            "-m",
            "FlagEmbedding.BGE_M3.run",
            "--output_dir",
            output_dir,
            "--model_name_or_path",
            self.path,
            "--train_data",
            output_dir,
            "--learning_rate",
            "1e-5",
            "--fp16",
            "--num_train_epochs",
            "5",
            "--per_device_train_batch_size",
            "32",
            "--dataloader_drop_last",
            "True",
            "--normlized",
            "True",
            "--temperature",
            "0.02",
            "--query_max_len",
            "64",
            "--passage_max_len",
            "256",
            "--train_group_size",
            "2",
            "--negatives_cross_device",
            "--logging_steps",
            "10",
            "--same_task_within_batch",
            "True",
            "--unified_finetuning",
            "True",
            "--use_self_distill",
            "True",
        ]

        run_shell_command(command)


class ColpaliRetriever(MultimodalRetriever):
    path: str = "vidore/colpali-v1.2"
    base: str = "vidore/colpaligemma-3b-pt-448-base"
    model: Optional[ColPali] = None
    processor: Optional[PaliGemmaProcessor] = None
    device: str = "cuda"
    cache: OrderedDict[str, list] = CollectionsOrderedDict()

    def load(self):
        if self.model is None:
            self.model = ColPali.from_pretrained(
                self.base, torch_dtype=torch.bfloat16, device_map=self.device
            )
            self.model.load_adapter(self.path)
            self.model = self.model.eval()
            self.processor = AutoProcessor.from_pretrained(self.path)

    def encode_document(self, doc: MultimodalDocument) -> List[torch.Tensor]:
        hash_id = hashlib.md5(doc.json().encode()).hexdigest()
        if len(self.cache) > 100:
            self.cache.popitem(last=False)
        if hash_id not in self.cache:
            images = [page.get_full_image() for page in doc.pages]
            # noinspection PyTypeChecker
            dataloader = DataLoader(
                images,
                batch_size=4,
                shuffle=False,
                collate_fn=lambda x: process_images(self.processor, x),
            )
            ds = []
            for batch_doc in tqdm(dataloader, desc="Encoding document"):
                with torch.no_grad():
                    batch_doc = {k: v.to(self.device) for k, v in batch_doc.items()}
                    embeddings_doc = self.model(**batch_doc)
                ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
            self.cache[hash_id] = ds

        return self.cache[hash_id]

    @staticmethod
    def compute_scores(qs: List[torch.Tensor], ds: List[torch.Tensor]) -> List[float]:
        evaluator = CustomEvaluator(is_multi_vector=True)
        if evaluator.is_multi_vector:
            scores = evaluator.evaluate_colbert(qs, ds)
        else:
            scores = evaluator.evaluate_biencoder(qs, ds)

        assert scores.shape[0] == len(qs)
        return scores.to(torch.float32).cpu().squeeze().tolist()

    def run(self, query: str, doc: MultimodalDocument) -> MultimodalDocument:
        doc = doc.copy(deep=True)
        self.load()
        ds = self.encode_document(doc)

        # noinspection PyTypeChecker
        dataloader = DataLoader(
            [query],
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: process_queries(
                self.processor, x, Image.new("RGB", (448, 448), (255, 255, 255))
            ),
        )

        qs = []
        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {k: v.to(self.device) for k, v in batch_query.items()}
                embeddings_query = self.model(**batch_query)
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

        scores = self.compute_scores(qs, ds)
        assert len(scores) == len(doc.pages)
        for i, page in enumerate(doc.pages):
            page.score = float(scores[i])

        return doc


class HybridRetriever(MultimodalRetriever):
    # Use Reciprocal Rank Fusion (RRF) scores to combine multiple retrievers
    models: List[MultimodalRetriever] = [BGEM3Retriever(), ColpaliRetriever()]
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
    elif name == "bge_finetune":
        return BGEM3Retriever(path="outputs/finetune/bge")
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


def run_finetune(name: str, data_path: str, output_dir: str, **kwargs):
    model = select_retriever(name, **kwargs)
    data = MultimodalData.load(data_path)
    model.finetune(data, output_dir)


"""
p retrieval.py run_finetune bge data/questions/train.json outputs/finetune/bge
"""

if __name__ == "__main__":
    Fire()
