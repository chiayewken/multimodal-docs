import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Union, Tuple

from PIL import Image
from fire import Fire
from tqdm import tqdm

from data_loading import (
    MultimodalDocument,
    MultimodalSample,
    MultimodalObject,
    MultimodalPage,
)
from modeling import select_model, EvalModel


def check_is_question_valid(
    question: str,
    target: List[Union[str, Image.Image]],
    category: str,
    context: List[Union[str, Image.Image]],
    model: EvalModel,
) -> Tuple[bool, str]:
    instruction = "\n".join(
        [
            f"You are given a document context and a question based on the target {category}. Your task is to determine whether the question is valid or invalid.",
            f"1. If the target does not contain any {category}, it is invalid."
            f"2. If the question does not address the {category}, it is invalid."
            f"3. If the question is confusing or cannot be answered, it is invalid."
            f"4. If the question is trivial or can be answered without the {category}, it is invalid."
            "Give a brief analysis of the question and then output either <valid> or <invalid>.",
        ]
    )

    inputs: List[Union[str, Image.Image]] = [
        instruction,
        "Document context:",
        *context,
        f"Question to be checked: '{question}'",
        f"Target {category}:",
        *target,
    ]

    output = model.run(inputs)
    info = dict(
        question=question,
        category=category,
        context=str([type(o) for o in context]),
        target=str([type(o) for o in target]),
        output=output,
    )
    print(json.dumps(info, indent=2))
    return "<valid>" in output, output


def prepare_target_and_context(
    page: MultimodalPage, doc: MultimodalDocument, category: str
) -> Tuple[List[Union[str, Image.Image]], List[Union[str, Image.Image]]]:
    if "text" in category.lower():
        target = [page.text]
    else:
        target = [o.get_image() for o in page.objects if o.category == category]

    context = []
    for other in doc.pages:
        if abs(other.number - page.number) <= 1:
            for o in other.get_tables_and_figures():
                context.append(o.get_image())
            if other.text:
                context.append(other.text)

    return target, context


def generate_questions(
    *paths: str,
    path_out: str,
    questions_per_doc: int = 15,
    object_categories: List[str] = ("Picture", "Table", "Text"),
    model_names: List[str] = (
        "gpt-4o-2024-05-13",
        "claude-3-5-sonnet-20240620",
        "gemini-1.5-pro-001",
    ),
    random_seed: int = 0,
):
    print(locals())
    instruction = "Based on this document, can you generate a challenging question that requires reasoning over the multimodal content of texts, figures, or tables? Output the question only."
    random.seed(random_seed)
    Path(path_out).parent.mkdir(exist_ok=True, parents=True)
    valid_counts = []
    category_counts = []

    with open(path_out, "w") as f:
        for doc_path in tqdm(random.sample(paths, k=len(paths)), desc=path_out):
            doc = MultimodalDocument.load(doc_path)
            o: MultimodalObject
            p: MultimodalPage

            for label in object_categories:
                samples = []
                for p in random.sample(doc.pages, len(doc.pages)):
                    if label not in [o.category for o in p.objects]:
                        continue
                    if len(samples) >= questions_per_doc // len(object_categories):
                        break
                    target, context = prepare_target_and_context(p, doc, label)
                    if target == [] or context == []:
                        continue

                    name = random.choice(model_names)
                    model = select_model(name)
                    mapping = dict(Table="tables", Picture="figures", Text="texts")
                    instruction = f"Based on the target {mapping[label]} in this document, can you generate a challenging question? Output the question only."
                    inputs = [
                        instruction,
                        "Document context:",
                        *context,
                        f"Target {mapping[label]}:",
                        *target,
                    ]
                    question: str = model.run(inputs).strip()

                    is_valid, checking_output = check_is_question_valid(
                        question, target, mapping[label], context, model
                    )
                    if is_valid:
                        answer = model.run([question, *inputs[1:]]).strip()
                        samples.append(
                            MultimodalSample(
                                question=question,
                                answer=answer,
                                category=mapping[label],
                                checking_output=checking_output,
                                evidence_pages=[p.number],
                                source=p.source,
                                annotator=name,
                            )
                        )
                        print(samples[-1].model_dump_json(indent=2))
                        print(samples[-1].model_dump_json(), file=f)
                        valid_counts.append(1)
                        category_counts.append(mapping[label])
                        print(dict(category_counts=Counter(category_counts)))
                    else:
                        valid_counts.append(0)
                    print(dict(valid_counts=sum(valid_counts) / len(valid_counts)))


"""
p question_generation.py generate_questions data/test/*.json --path_out data/questions/test.json
"""


if __name__ == "__main__":
    Fire()
