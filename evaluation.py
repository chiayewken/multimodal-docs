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


def test_retriever(*data_paths: str, retriever_name: str, path: str = "retrieve.json"):
    retriever = select_retriever(retriever_name)
    samples = []
    for dp in data_paths:
        samples.extend(MultimodalData.load(dp).samples)
    data = MultimodalData(samples=samples)
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
        "azure",
        "claude-3-5-sonnet-20240620",
        "gemini-1.5-pro-002",
    ),
    use_answer: bool = False,
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
            if use_answer:
                sample.pred = sample.answer
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

bash scripts/eval_retrievers.sh 
p analysis.py test_retriever_results outputs/retrieve/test/*.json
python evaluation.py test_retriever data/questions/train.json --retriever_name colpali --path outputs/retrieve/train/colpali.json
python evaluation.py test_retriever data/questions/train2.json --retriever_name colpali --path outputs/retrieve/train2/colpali.json
python evaluation.py test_retriever data/questions/train3.json --retriever_name colpali --path outputs/retrieve/train3/colpali.json
python evaluation.py test_retriever data/questions/train4.json --retriever_name colpali --path outputs/retrieve/train4/colpali.json

                                 path  text  figure  table   all
1     outputs/retrieve/test/bm25.json  52.8    38.1   43.4  44.9
2     outputs/retrieve/test/clip.json  61.0    42.1   50.1  51.2
0      outputs/retrieve/test/bge.json  69.3    49.5   64.0  61.2
3  outputs/retrieve/test/colpali.json  73.5    62.7   73.7  70.1

p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name azure
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name claude-3-5-sonnet-20240620
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name gemini-1.5-pro-002
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name qwen
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name onevision
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name pixtral
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name custom_qwen

p evaluation.py run_multi_judge outputs/azure/colpali/top_k=5.json
p evaluation.py run_multi_judge outputs/claude-3-5-sonnet-20240620/colpali/top_k=5.json
p evaluation.py run_multi_judge outputs/gemini-1.5-pro-002/colpali/top_k=5.json
p evaluation.py run_multi_judge outputs/qwen/colpali/top_k=5.json
p evaluation.py run_multi_judge outputs/pixtral/colpali/top_k=5.json
p evaluation.py run_multi_judge outputs/onevision/colpali/top_k=5.json
p evaluation.py run_multi_judge outputs/custom_qwen/colpali/top_k=5.json
p evaluation.py run_multi_judge outputs/retrieve/train/colpali_copy.json --use_answer
p evaluation.py run_multi_judge outputs/swift_qwen/colpali/top_k=5.json
p evaluation.py run_multi_judge outputs/swift_qwen_10k/colpali/top_k=5.json

p analysis.py test_results outputs/*/colpali/top_k=5.json

                                                path  text  figure  table   all
0             outputs/onevision/colpali/top_k=5.json  3.95    3.53   3.29  3.59
1                  outputs/qwen/colpali/top_k=5.json  4.00    3.80   3.62  3.81
2               outputs/pixtral/colpali/top_k=5.json  4.30    4.16   4.11  4.19
3                 outputs/azure/colpali/top_k=5.json  4.47    4.33   4.51  4.44
4  outputs/claude-3-5-sonnet-20240620/colpali/top...  4.52    4.38   4.56  4.49
5    outputs/gemini-1.5-pro-002/colpali/top_k=5.json  4.53    4.39   4.55  4.49

p analysis.py test_results outputs/*/colpali/top_k=5.json --valid_path data/annotation/valid_questions.json

                         path  text  figure  table   all
0                   onevision  4.03    3.57   3.30  3.62
1                 custom_qwen  4.07    3.84   3.49  3.79
2                        qwen  4.08    3.83   3.62  3.84
3                     pixtral  4.38    4.20   4.09  4.22
4                       azure  4.55    4.38   4.53  4.49
5  claude-3-5-sonnet-20240620  4.57    4.42   4.54  4.51
6          gemini-1.5-pro-002  4.59    4.43   4.52  4.51

################################################################################
Analysis on different top-k values (100 test samples)

p data_loading.py sample_questions outputs/retrieve/test/colpali.json outputs/retrieve/test/colpali_sample_100.json --num 100
p evaluation.py generate_answers outputs/retrieve/test/colpali_sample_100.json --retriever_name colpali_sample_100 --generator_name gemini-1.5-pro-002 --top_k 1
p evaluation.py generate_answers outputs/retrieve/test/colpali_sample_100.json --retriever_name colpali_sample_100 --generator_name gemini-1.5-pro-002 --top_k 5
p evaluation.py generate_answers outputs/retrieve/test/colpali_sample_100.json --retriever_name colpali_sample_100 --generator_name gemini-1.5-pro-002 --top_k 10
p evaluation.py generate_answers outputs/retrieve/test/colpali_sample_100.json --retriever_name colpali_sample_100 --generator_name gemini-1.5-pro-002 --top_k 20
p evaluation.py run_multi_judge outputs/gemini-1.5-pro-002/colpali_sample_100/top_k=1.json
p evaluation.py run_multi_judge outputs/gemini-1.5-pro-002/colpali_sample_100/top_k=5.json
p evaluation.py run_multi_judge outputs/gemini-1.5-pro-002/colpali_sample_100/top_k=10.json
p evaluation.py run_multi_judge outputs/gemini-1.5-pro-002/colpali_sample_100/top_k=20.json

p evaluation.py generate_answers outputs/retrieve/test/colpali_sample_100.json --retriever_name colpali_sample_100 --generator_name qwen --top_k 1
p evaluation.py generate_answers outputs/retrieve/test/colpali_sample_100.json --retriever_name colpali_sample_100 --generator_name qwen --top_k 5
p evaluation.py generate_answers outputs/retrieve/test/colpali_sample_100.json --retriever_name colpali_sample_100 --generator_name qwen --top_k 10
p evaluation.py generate_answers outputs/retrieve/test/colpali_sample_100.json --retriever_name colpali_sample_100 --generator_name qwen --top_k 20
p evaluation.py run_multi_judge outputs/qwen/colpali_sample_100/top_k=1.json
p evaluation.py run_multi_judge outputs/qwen/colpali_sample_100/top_k=5.json
p evaluation.py run_multi_judge outputs/qwen/colpali_sample_100/top_k=10.json
p evaluation.py run_multi_judge outputs/qwen/colpali_sample_100/top_k=20.json

p analysis.py test_results outputs/*/colpali_sample_100/top_k=*.json --sort_key "path"

################################################################################
Input the full high-res image intead of extracted texts and table / figure images (to be updated)

p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name claude-3-5-sonnet-20240620 --use_full_image
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name gemini-1.5-pro-002 --use_full_image
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name highres_onevision --use_full_image
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name highres_intern --use_full_image
p evaluation.py generate_answers outputs/retrieve/test/colpali.json --retriever_name colpali --generator_name highres_idefics --use_full_image

p evaluation.py run_multi_judge outputs/claude-3-5-sonnet-20240620/colpali/top_k=5_full_image.json
p evaluation.py run_multi_judge outputs/gemini-1.5-pro-002/colpali/top_k=5_full_image.json
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
