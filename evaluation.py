from pathlib import Path
from typing import Tuple

import pandas as pd
from fire import Fire
from tqdm import tqdm

from data_loading import (
    MultimodalData,
    MultimodalObject,
    MultimodalDocument,
    MultimodalSample,
)
from modeling import select_model
from retrieval import select_retriever


def safe_divide(a: float, b: float) -> float:
    return a / b if b != 0 else 0


def test_retriever(data_path: str, retriever_name: str, top_k: int = 5):
    retriever = select_retriever(retriever_name, top_k=top_k)
    data = MultimodalData.load(data_path)
    scores = []

    def score_fn(s: MultimodalSample) -> Tuple[float, float, float]:
        num_correct = len(set(s.evidence_pages) & set(s.retrieved_pages))
        precision = safe_divide(num_correct, len(s.retrieved_pages))
        recall = safe_divide(num_correct, len(s.evidence_pages))
        f1 = safe_divide(2 * precision * recall, (precision + recall))
        return precision, recall, f1

    for sample in tqdm(data.samples, desc=retriever_name):
        doc = MultimodalDocument.load(sample.source)
        query = MultimodalObject(text=sample.question)
        pages = retriever.run(query, doc).objects
        sample.retrieved_pages = [p.page for p in pages]
        scores.append(score_fn(sample))

    df = pd.DataFrame(scores, columns=["precision", "recall", "f1"])
    df = df.apply(lambda x: x * 100)
    print(df.mean(axis=0).round(2).to_dict())


def generate_answers(
    data_path: str,
    generator_name: str,
    retriever_name: str,
    output_dir: str = "outputs",
    top_k: int = 5,
):
    generator = select_model(generator_name)
    retriever = select_retriever(retriever_name, top_k=top_k)
    data = MultimodalData.load(data_path)
    path_out = Path(output_dir, generator_name, retriever_name, f"{top_k=}.json")
    Path(path_out).parent.mkdir(exist_ok=True, parents=True)

    with open(path_out, "w") as f:
        for sample in tqdm(data.samples, desc=str(path_out)):
            doc = MultimodalDocument.load(sample.source)
            query = MultimodalObject(text=sample.question)
            pages = retriever.run(query, doc).objects
            sample.retrieved_pages = [p.page for p in pages]

            prompt = f"Answer the following question in 200 words or less: {query.text}"
            inputs = [prompt] + [p.get_image() or p.text for p in pages]
            sample.pred = generator.run(inputs)
            sample.generator = generator_name
            print(sample.json(indent=2))
            print(sample.json(), file=f)


"""
p evaluation.py test_retriever data/questions.json --retriever_name bm25
{'precision': 28.6, 'recall': 28.73, 'f1': 28.65}

p evaluation.py test_retriever data/questions.json --retriever_name colpali
{'precision': 39.0, 'recall': 39.27, 'f1': 39.1}

p evaluation.py generate_answers data/questions.json --generator_name openai --retriever_name bm25
"""


if __name__ == "__main__":
    Fire()
