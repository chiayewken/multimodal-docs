import re
from pathlib import Path
from typing import List, Union

from PIL import Image
from fire import Fire
from tqdm import tqdm

from data_loading import MultimodalData, MultimodalDocument, Judgement
from modeling import select_model
from retrieval import select_retriever


def safe_divide(a: float, b: float) -> float:
    return a / b if b != 0 else 0


def test_retriever(data_path: str, retriever_name: str, path: str = "retrieve.json"):
    retriever = select_retriever(retriever_name)
    data = MultimodalData.load(data_path)
    doc_map = {}
    scores = []

    progress = tqdm(data.samples, desc=retriever_name)
    for sample in progress:
        if sample.source not in doc_map:
            doc_map[sample.source] = MultimodalDocument.load(sample.source)
        doc = doc_map[sample.source]
        output = retriever.run(sample.question, doc)

        # Calculate MRR score
        sorted_pages = sorted(output.pages, key=lambda p: p.score, reverse=True)
        sorted_ids = [p.number for p in sorted_pages]
        assert len(sample.evidence_pages) == 1
        rank = sorted_ids.index(sample.evidence_pages[0])
        scores.append(1 / (rank + 1))
        progress.set_postfix(score=sum(scores) / len(scores))
        sample.retrieved_pages = sorted_ids

    data.save(path)
    print(Path(path).absolute())
    return sum(scores) / len(scores)


def generate_answers(
    data_path: str,
    generator_name: str,
    retriever_name: str,
    output_dir: str = "outputs",
    top_k: int = 5,
    use_full_image: bool = False,
):
    generator = select_model(generator_name)
    data = MultimodalData.load(data_path)
    path_out = Path(output_dir, generator_name, retriever_name, f"{top_k=}.json")
    Path(path_out).parent.mkdir(exist_ok=True, parents=True)
    if use_full_image:
        path_out = path_out.with_name(
            path_out.name.replace(".json", "_full_image.json")
        )

    with open(path_out, "w") as f:
        for sample in tqdm(data.samples, desc=str(path_out)):
            doc = MultimodalDocument.load(sample.source)
            assert sample.retrieved_pages
            sample.retrieved_pages = sorted(sample.retrieved_pages[:top_k])

            context = []
            for p in doc.pages:
                if p.number in sample.retrieved_pages:
                    if p.text:
                        context.append(p.text)
                    context.extend(o.get_image() for o in p.get_tables_and_figures())

            if use_full_image:
                context = [
                    p.get_full_image()
                    for p in doc.pages
                    if p.number in sample.retrieved_pages
                ]

            inputs = [
                "Context:",
                *context,
                f"Answer the following question in 200 words or less: {sample.question}",
            ]
            sample.pred = generator.run(inputs)
            sample.generator = generator_name
            print(sample.model_dump_json(indent=2))
            print(sample.model_dump_json(), file=f)


def extract_score(text) -> int:
    matches = re.findall(r"\b(\d+)\b", text)
    if matches:
        return int(matches[-1])
    return -1


def prepare_document_context(
    page_number: int, doc: MultimodalDocument
) -> List[Union[str, Image.Image]]:
    context = []
    for other in doc.pages:
        if abs(other.number - page_number) <= 1:
            if other.number == page_number:
                for o in other.get_tables_and_figures():
                    context.append(o.get_image())
            if other.text:
                context.append(other.text)

    return context


def run_multi_judge(
    data_path: str,
    judge_names: List[str] = (
        "gpt-4o-2024-08-06",
        "claude-3-5-sonnet-20240620",
        "gemini-1.5-pro-001",
    ),
):
    parts = [
        "Instruction: You will be given one response to a question based on the multimodal document containing texts, figures, or tables. Your task is to rate the response on one metric. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.",
        "Evaluation Criteria: Correctness (1-5) refers to the degree to which the response accurately, comprehensively, and appropriately addresses the question based on the information provided in the document.",
        "5 - Fully Correct: The response is completely accurate, comprehensively addresses the question, fully integrates relevant information from all parts of the document, and provides a coherent answer.",
        "4 - Mostly Correct: The response is largely accurate with only minor errors or omissions, addresses most main points, and integrates information well from the document.",
        "3 - Partially Correct: The response contains a mix of accurate and inaccurate information, addresses some key points but misses others, and partially integrates information from the document.",
        "2 - Mostly Incorrect: The response has multiple inaccuracies, addresses only a small portion correctly, and shows minimal integration of information from the document.",
        "1 - Completely Incorrect: The response contains significant errors, is irrelevant, or fails to address the question based on the document.",
        "Evaluation Steps:",
        "- Thoroughly review the multimodal document (text, figures, tables) and the question.",
        "- Carefully read the response, comparing it to the information in the document.",
        "- Assess the response's accuracy, comprehensiveness, and relevance to the question.",
        "- Assign a correctness score from 1 to 5 based on the Evaluation Criteria provided.",
        "Question: {question}",
        "Response: {answer}",
        "Evaluation Form (score only without explanation)\nCorrectness:",
    ]
    template = "\n".join(parts)

    data = MultimodalData.load(data_path)
    progress = tqdm(data.samples, desc=data_path)
    scores = []

    for sample in progress:
        judgements = []
        for name in judge_names:
            judge = select_model(name)
            doc = MultimodalDocument.load(sample.source)
            assert len(sample.evidence_pages) == 1
            context = prepare_document_context(sample.evidence_pages[0], doc)
            instruction = template.format(question=sample.question, answer=sample.pred)
            inputs = ["Document:", *context, instruction]

            outputs = judge.run(inputs)
            scores.append(extract_score(outputs))
            judgements.append(Judgement(name=name, content=outputs, score=scores[-1]))

        sample.judgements = judgements
        print(sample.model_dump_json(indent=2))
        progress.set_postfix(score=sum(scores) / len(scores))
    data.save(data_path)


"""
# Evaluate different retrieval methods

python evaluation.py test_retriever data/questions/train.json --retriever_name bm25
[00:22<00:00, 25.11it/s, score=0.386]
python evaluation.py test_retriever data/questions/train.json --retriever_name clip
[05:51<00:00,  1.57it/s, score=0.443]
python evaluation.py test_retriever data/questions/train.json --retriever_name bge
[03:21<00:00,  2.75it/s, score=0.526]
python evaluation.py test_retriever data/questions/train.json --retriever_name bge_finetune
[03:44<00:00,  2.46it/s, score=0.636]
python evaluation.py test_retriever data/questions/train.json --retriever_name colpali
[14:10<00:00,  1.54s/it, score=0.625]
python evaluation.py test_retriever data/questions/train.json --retriever_name hybrid
[16:39<00:00,  1.81s/it, score=0.588]

# Retrieve first and generate answers and evaluate with multi-judge

python evaluation.py test_retriever data/questions/test.json --retriever_name colpali --path outputs/retrieve/test/colpali.json
90/90 [16:28<00:00, 10.98s/it, score=0.674]

p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name gpt-4o-2024-08-06
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name claude-3-5-sonnet-20240620
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name gemini-1.5-pro-001
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name intern
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name idefics
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name onevision
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name gpt-4o-mini-2024-07-18

p evaluation.py run_multi_judge outputs/gpt-4o-2024-08-06/colpali/top_k=5.json
p evaluation.py run_multi_judge outputs/claude-3-5-sonnet-20240620/colpali/top_k=5.json
p evaluation.py run_multi_judge outputs/gemini-1.5-pro-001/colpali/top_k=5.json
p evaluation.py run_multi_judge outputs/intern/colpali/top_k=5.json
p evaluation.py run_multi_judge outputs/idefics/colpali/top_k=5.json
p evaluation.py run_multi_judge outputs/onevision/colpali/top_k=5.json
p evaluation.py run_multi_judge outputs/gpt-4o-mini-2024-07-18/colpali/top_k=5.json

p analysis.py test_results outputs/*/colpali/top_k=5.json
                                                path  text  figure  table   all
0  outputs/claude-3-5-sonnet-20240620/colpali/top...  4.68    4.27   4.64  4.53
1    outputs/gemini-1.5-pro-001/colpali/top_k=5.json  4.63    4.19   4.32  4.38
2     outputs/gpt-4o-2024-08-06/colpali/top_k=5.json  4.72    4.41   4.48  4.54
3  outputs/gpt-4o-mini-2024-07-18/colpali/top_k=5...  4.60    3.90   4.11  4.20
4               outputs/idefics/colpali/top_k=5.json  4.02    2.64   2.59  3.09
5                outputs/intern/colpali/top_k=5.json  4.38    3.70   3.52  3.87
6             outputs/onevision/colpali/top_k=5.json  4.27    3.66   3.47  3.80

################################################################################
Input the full high-res image intead of extracted texts and table / figure images

p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name gpt-4o-2024-08-06 --use_full_image
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name claude-3-5-sonnet-20240620 --use_full_image
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name gemini-1.5-pro-001 --use_full_image
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name highres_onevision --use_full_image
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name highres_intern --use_full_image
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name highres_idefics --use_full_image

p evaluation.py run_multi_judge outputs/gpt-4o-2024-08-06/colpali/top_k=5_full_image.json
p evaluation.py run_multi_judge outputs/claude-3-5-sonnet-20240620/colpali/top_k=5_full_image.json
p evaluation.py run_multi_judge outputs/gemini-1.5-pro-001/colpali/top_k=5_full_image.json
p evaluation.py run_multi_judge outputs/highres_onevision/colpali/top_k=5_full_image.json
p evaluation.py run_multi_judge outputs/highres_intern/colpali/top_k=5_full_image.json
p evaluation.py run_multi_judge outputs/highres_idefics/colpali/top_k=5_full_image.json

p analysis.py test_results outputs/*/colpali/top_k=5_full_image.json
                                                path  text  figure  table   all
0  outputs/claude-3-5-sonnet-20240620/colpali/top...  4.73    4.34   4.52  4.53
1  outputs/gemini-1.5-pro-001/colpali/top_k=5_ful...  4.09    4.02   3.46  3.86
2  outputs/gpt-4o-2024-08-06/colpali/top_k=5_full...  4.61    4.17   4.43  4.40
3  outputs/highres_idefics/colpali/top_k=5_full_i...  2.80    2.69   2.37  2.62
4  outputs/highres_intern/colpali/top_k=5_full_im...  3.74    3.23   2.88  3.29
5  outputs/highres_onevision/colpali/top_k=5_full...  3.31    2.93   3.07  3.10

"""


if __name__ == "__main__":
    Fire()
