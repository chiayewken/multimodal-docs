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


def check_question(
    question: str,
    target: List[Union[str, Image.Image]],
    context: str,
    category: str,
    model: EvalModel,
) -> bool:
    checks = [
        f"Based on the document content and question, answer yes or no only: Does the content contain any {category}?",
        f"Based on the document content and question, answer yes or no only: Is the question relevant to the {category}?",
        f"Based on the document content and question, answer yes or no only: Are the {category} necessary in order to answer the question?",
        f"Based on the document content and question, answer yes or no only: Is the question clear and answerable based on the {category}?",
    ]

    for instruction in checks:
        inputs: List[Union[str, Image.Image]] = [
            instruction,
            f"Context: {context}",
            f"{category.capitalize()}:",
            *target,
            f"Question to be checked: '{question}'",
        ]

        output = model.run(inputs)
        info = dict(
            question=question,
            category=category,
            instruction=instruction,
            output=output,
        )

        print(json.dumps(info, indent=2))
        pred = "yes" if "yes" in output.lower() else "no"
        if pred == "no":
            return False

    return True


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
                    mapping = dict(
                        Table="tables",
                        Picture="figures or diagrams or charts",
                        Text="texts",
                    )
                    instruction = f"Based on the target {mapping[label]} in this document, can you generate a challenging question? Output the question only."
                    inputs = [
                        instruction,
                        "Document context:",
                        *context,
                        f"Target {mapping[label]}:",
                        *target,
                    ]

                    question: str = model.run(inputs).strip()
                    print(dict(doc=p.source, page=p.number, question=question))
                    is_valid = check_question(
                        question, target, p.text, mapping[label], model
                    )

                    if is_valid:
                        samples.append(
                            MultimodalSample(
                                question=question,
                                answer="",
                                category=mapping[label],
                                evidence_pages=[p.number],
                                source=doc_path,
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
