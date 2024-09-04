import re
from pathlib import Path
from typing import List

from fire import Fire
from tqdm import tqdm

from data_loading import (
    MultimodalData,
    MultimodalObject,
    MultimodalDocument,
    Judgement,
)
from modeling import select_model
from retrieval import select_retriever


def safe_divide(a: float, b: float) -> float:
    return a / b if b != 0 else 0


def test_retriever(data_path: str, retriever_name: str):
    retriever = select_retriever(retriever_name)
    data = MultimodalData.load(data_path)
    scores = []

    progress = tqdm(data.samples, desc=retriever_name)
    for sample in progress:
        doc = MultimodalDocument.load(sample.source)
        output = retriever.run(sample.question, doc)

        # Calculate MRR score
        sorted_pages = sorted(output.pages, key=lambda p: p.score, reverse=True)
        sorted_ids = [p.number for p in sorted_pages]
        assert len(sample.evidence_pages) == 1
        rank = sorted_ids.index(sample.evidence_pages[0])
        scores.append(1 / (rank + 1))
        progress.set_postfix(score=sum(scores) / len(scores))
        sample.retrieved_pages = sorted_ids

    data.save(data_path)
    return sum(scores) / len(scores)


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
            print(sample.model_dump_json(indent=2))
            print(sample.model_dump_json(), file=f)


def extract_score(text) -> int:
    for match in re.findall(r"\*\*(\d+)\*\*", text):
        return int(match)
    return -1


def run_multi_judge(
    data_path: str, judge_names: List[str] = ("openai", "claude", "gemini")
):
    template = "Score the following answer to the question based on the document content on a continuous scale from 0 to 100."
    template += '\nA score of zero means "irrelevant or missing answer that is not helpful at all" and score of one hundred means "perfect answer, well-grounded in the document with explanation that correctly answers the question".'
    template += "\nPlease provide a concise explanation of 1-3 sentences and then score from 0 to 100 that reflects the quality of the answer, not the quality of the question."
    template += "\nPlease follow this example output format: The answer correct identifies the weighted average and explains the trend, but there is a factual error. Thus, the score is **75**."
    template += "\n\nQuestion: {question}"
    template += "\n\nAnswer: {answer}"
    template += "\n\nDocument:"

    data = MultimodalData.load(data_path)
    progress = tqdm(data.samples, desc=f"{data_path} ({judge_names})")
    scores = []

    for sample in progress:
        judgements = []
        for name in judge_names:
            judge = select_model(name)
            prompt = template.format(question=sample.question, answer=sample.pred)
            doc = MultimodalDocument.load(sample.source)
            page_ids = sample.evidence_pages + sample.retrieved_pages
            pages = sorted(
                [p for p in doc.objects if p.page in page_ids], key=lambda p: p.page
            )

            inputs = [prompt] + [p.get_image() or p.text for p in pages] + ["\nScore:"]
            outputs = judge.run(inputs)
            scores.append(extract_score(outputs))
            judgements.append(Judgement(name=name, content=outputs, score=scores[-1]))

        sample.judgements = judgements
        print(sample.model_dump_json(indent=2))
        progress.set_postfix(score=sum(scores) / len(scores))
    data.save(data_path)


"""
# Evaluate different retrieval methods

python evaluation.py test_retriever data/questions/test.json --retriever_name bm25
[01:49<00:00,  2.45it/s, score=0.547]
python evaluation.py test_retriever data/questions/test.json --retriever_name colpali
[29:59<00:00,  6.69s/it, score=0.716]
python evaluation.py test_retriever data/questions/test.json --retriever_name clip
[28:10<00:00,  6.28s/it, score=0.487]
python evaluation.py test_retriever data/questions/test.json --retriever_name bge
[03:34<00:00,  1.25it/s, score=0.646]
python evaluation.py test_retriever data/questions/test.json --retriever_name hybrid
[13:09<00:00, 13.62s/it, score=0.751]

# Generate answers and evaluate with multi-judge

p evaluation.py generate_answers data/questions.json --generator_name openai --retriever_name bm25
p evaluation.py generate_answers data/questions.json --generator_name openai --retriever_name colpali
p evaluation.py generate_answers data/questions.json --generator_name gemma --retriever_name colpali
p evaluation.py generate_answers data/questions.json --generator_name idefics --retriever_name colpali
p evaluation.py generate_answers data/questions.json --generator_name idefics --retriever_name bm25
p evaluation.py generate_answers data/questions.json --generator_name claude --retriever_name colpali
p evaluation.py generate_answers data/questions.json --generator_name gemini --retriever_name colpali
p evaluation.py generate_answers data/questions.json --generator_name openai_mini --retriever_name colpali
p evaluation.py generate_answers data/questions.json --generator_name gemini_flash --retriever_name colpali

p evaluation.py run_multi_judge "outputs/openai/colpali/top_k=5.json"
[43:33<00:00, 26.14s/it, score=78.5]

p evaluation.py run_multi_judge "outputs/claude/colpali/top_k=5.json"
[43:33<00:00, 26.14s/it, score=79.5]

p evaluation.py run_multi_judge "outputs/gemini/colpali/top_k=5.json"
[43:24<00:00, 26.05s/it, score=75.3]

"""


if __name__ == "__main__":
    Fire()
