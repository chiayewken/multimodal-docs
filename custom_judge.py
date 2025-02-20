import json
import statistics
from pathlib import Path
from typing import List, Union

from PIL.Image import Image
from fire import Fire
from tqdm import tqdm

from data_loading import MultimodalData, MultimodalDocument, resize_image, save_image
from evaluation import prepare_document_context


def make_judge_instruction(
    question: str,
    response: str,
    doc: MultimodalDocument,
    page_number: int,
) -> List[Union[Image, str]]:
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
    context = prepare_document_context(page_number, doc)
    instruction = template.format(question=question, answer=response)
    return ["Document:", *context, instruction]


def make_train_data(*paths: str, path_out: str, image_dir: str = "data/judge_images"):
    # Adapted from evaluation.py run_multi_judge
    documents = {}
    pairs = []

    for p in paths:
        data = MultimodalData.load(p)
        if len(data.samples) < 1000:
            print(dict(skip=p))
            continue
        for sample in tqdm(data.samples):
            if sample.source not in documents:
                documents[sample.source] = MultimodalDocument.load(sample.source)

            doc = documents[sample.source]
            assert len(sample.evidence_pages) == 1
            inputs = make_judge_instruction(
                sample.question, sample.pred, doc, sample.evidence_pages[0]
            )

            # Get mode or median
            scores = [j.score for j in sample.judgements]
            if len(scores) == 3:
                score = statistics.median(scores)
                pairs.append((inputs, str(score)))

    print(dict(pairs=len(pairs)))
    Path(path_out).parent.mkdir(parents=True, exist_ok=True)
    with open(path_out, "w") as f:
        for inputs, response in tqdm(pairs, desc=str(path_out)):
            # Follow make_qwen_train_inputs
            images = [resize_image(x, 768) for x in inputs if isinstance(x, Image)]
            text = "\n\n".join([x for x in inputs if isinstance(x, str)])
            for x in inputs:
                if isinstance(x, Image):
                    text = "<image>" + text

            image_paths = [save_image(x, image_dir) for x in images]
            info = dict(query=text, response=response, images=image_paths)
            print(json.dumps(info), file=f)
            print(json.dumps(info))


def run_judge_evaluation():
    pass


"""
p custom_judge.py make_train_data outputs/*/colpali/top_k=5.json --path_out data/judge/train.jsonl

swift sft \
--rescale_image 240000 \
--max_length 6144 \
--lora_rank 64 \
--model_type qwen2-vl-7b-instruct \
--sft_type lora \
--dataset data/judge/train.jsonl

"""


if __name__ == "__main__":
    Fire()
