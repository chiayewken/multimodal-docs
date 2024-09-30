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
    target: Union[str, Image.Image],
    context: str,
    category: str,
    model: EvalModel,
) -> bool:
    parts = [
        f"Based on the document content and question, answer yes or no only to the following questions:",
        f"1. Does the question require information from the {category}?",
        f"2. Is the question clear and answerable based on the {category}?",
        f"3. Is the question of reasonable difficulty and answer cannot be simply copied?",
        f"Give the response in the following example format: 1. yes/no 2. yes/no 3. yes/no",
    ]

    checks = ["\n".join(parts)]  # Combine multiple checks into one prompt / response
    for instruction in checks:
        inputs: List[Union[str, Image.Image]] = [
            f"Document context:\n\n{context}",
            f"{category.capitalize()}:",
            target,
            f"Question to be checked: '{question}'",
            f"Instruction: {instruction}",
        ]

        output = model.run(inputs)
        labels = [word for word in output.lower().split() if word in ["yes", "no"]]
        info = dict(
            question=question,
            category=category,
            instruction=instruction,
            output=output,
            valid=all(x == "yes" for x in labels),
        )

        print(json.dumps(info, indent=2))
        if not all(x == "yes" for x in labels):
            return False

    return True


def check_target(x: Union[str, Image.Image], category: str, model: EvalModel) -> bool:
    if isinstance(x, str):
        assert "text" in category.lower()
        return True
    elif "figure" in category.lower():
        instruction = f"Is this image a valid example of a {category.lower()}? Note that equations are not valid. Answer yes or no only."
    elif "table" in category.lower():
        instruction = f"Is this image a valid example of a {category.lower()}? Note that table of contents are not valid. Answer yes or no only."
    else:
        raise ValueError

    assert isinstance(x, Image.Image)
    output = model.run([instruction, x])
    print(dict(target=x, category=category, instruction=instruction, output=output))
    return "yes" in output.lower()


def prepare_target_and_context(
    page: MultimodalPage, doc: MultimodalDocument, category: str
) -> Tuple[Union[str, Image.Image], str]:
    context = []
    for other in doc.pages:
        if abs(other.number - page.number) <= 1:
            if other.text:
                context.append(other.text)

    if "text" in category.lower():
        return page.text, "\n\n".join(context)
    else:
        images = [o.get_image() for o in page.objects if o.category == category]
        return random.choice(images), "\n\n".join(context)


def generate_answer(
    question: str,
    target: Union[str, Image.Image],
    context: str,
    category: str,
    model: EvalModel,
) -> str:
    inputs = [
        f"Context: {context}",
        f"Target {category.lower()}:",
        target,
        f"Based on the context and target {category.lower()}, answer the following question in 200 words or less: {question}",
    ]
    return model.run(inputs)


def generate_questions(
    *paths: str,
    path_out: str,
    questions_per_doc: int = 15,
    object_categories: List[str] = ("Picture", "Table", "Text"),
    model_names: List[str] = (
        # "gpt-4o-2024-08-06",
        "azure",
        "claude-3-5-sonnet-20240620",
        "gemini-1.5-pro-002",
    ),
    random_seed: int = 0,
    do_verify: bool = True,
    exclude: List[str] = (),
    use_answer: bool = False,
):
    print(locals())
    random.seed(random_seed)
    Path(path_out).parent.mkdir(exist_ok=True, parents=True)
    valid_counts = []
    category_counts = []
    model_map = {m: select_model(m) for m in model_names}
    language_checker = select_model("langdetect")

    lst = []
    for doc_path in paths:
        if any(Path(doc_path).name.startswith(str(prefix)) for prefix in exclude):
            continue
        lst.append(doc_path)
    paths = lst
    print(dict(paths=len(paths)))

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
                    if language_checker.run([p.text]).strip() != "en":
                        continue
                    target, context = prepare_target_and_context(p, doc, label)
                    if not context:
                        continue

                    name = random.choice(model_names)
                    model = model_map[name]
                    mapping = dict(
                        Table="tables",
                        Picture="figures or diagrams or charts",
                        Text="texts",
                    )
                    instruction = f"Based on the target {mapping[label]} in this document, can you generate a test question? Ensure that the question is challenging and the answer cannot be simply copied from the content. Output the question only."
                    inputs = [
                        f"Document context:\n\n{context}",
                        f"Target {mapping[label]}:",
                        target,
                        f"Instruction: {instruction}",
                    ]

                    if not check_target(target, mapping[label], model):
                        continue

                    question: str = model.run(inputs).strip()
                    print(dict(doc=p.source, page=p.number, question=question))
                    is_valid = not do_verify or check_question(
                        question, target, p.text, mapping[label], model
                    )

                    if is_valid:
                        samples.append(
                            MultimodalSample(
                                question=question,
                                answer=generate_answer(
                                    question, target, context, mapping[label], model
                                ),
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
p question_generation.py generate_questions data/train/*.json --path_out data/questions/train.json --questions_per_doc 30 --do_verify False
p question_generation.py generate_questions data/test/*.json --path_out data/questions/test.json --questions_per_doc 3
p question_generation.py generate_questions data/test/NYSE*.json --path_out data/questions/test_finance.json --questions_per_doc 6
p question_generation.py generate_questions data/test/24*.json --path_out data/questions/test_academic.json --questions_per_doc 6
p question_generation.py generate_questions data/test/*.json --exclude "24,NYSE" --path_out data/questions/test_product.json --questions_per_doc 6
p question_generation.py generate_questions data/train/*.json --path_out data/questions/train.json --questions_per_doc 9 --use_answer
p question_generation.py generate_questions data/train/*.json --path_out data/questions/train2.json --questions_per_doc 9 --use_answer
p question_generation.py generate_questions data/train/*.json --path_out data/questions/train3.json --questions_per_doc 9 --use_answer --random_seed 3
p question_generation.py generate_questions data/train/*.json --path_out data/questions/train4.json --questions_per_doc 9 --use_answer --random_seed 4
"""


if __name__ == "__main__":
    Fire()
