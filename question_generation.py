import random
from pathlib import Path
from typing import List

from fire import Fire
from tqdm import tqdm

from data_loading import (
    MultimodalDocument,
    convert_text_to_image,
    MultimodalSample,
    MultimodalObject,
)
from modeling import select_model, EvalModel


def check_is_page_multimodal(pages: List[MultimodalObject], model: EvalModel) -> bool:
    instruction = "Check if the document contains multimodal content such as text, figures, and tables. If it contains only text or simple images, output <text>. Otherwise, output <multimodal>."
    inputs = [instruction] + [convert_text_to_image(p.image_string) for p in pages]
    output = model.run(inputs)
    label = "<multimodal>" if "<multimodal>" in output else "<text>"
    print(dict(page=[p.id for p in pages], page_label=label))
    return label == "<multimodal>"


def check_is_question_multimodal(
    question: str, pages: List[MultimodalObject], model: EvalModel
) -> bool:
    instruction = "Check if the question requires reasoning over the multimodal content of texts, figures, or tables. If the question requires only text-based reasoning, output <text>. Otherwise, output <multimodal>"
    inputs = [instruction, f"Question to be checked: '{question}'"]
    inputs.extend([convert_text_to_image(p.image_string) for p in pages])
    output = model.run(inputs)
    label = "<multimodal>" if "<multimodal>" in output else "<text>"
    print(dict(page=[p.id for p in pages], question_label=label))
    return label == "<multimodal>"


def generate_questions(
    *paths: str,
    path_out: str,
    questions_per_doc: int = 10,
    model_names: List[str] = ("openai", "claude", "gemini"),
    random_seed: int = 0,
):
    print(locals())
    instruction = "Based on this document, can you generate a challenging question that requires reasoning over the multimodal content of texts, figures, or tables? Output the question only."
    random.seed(random_seed)
    Path(path_out).parent.mkdir(exist_ok=True, parents=True)

    with open(path_out, "w") as f:
        for p in tqdm(paths, desc=path_out):
            samples = []
            doc = MultimodalDocument.load(p)
            o: MultimodalObject
            for o in random.sample(doc.objects, len(doc.objects)):
                pages = [b for b in doc.objects if o.page - 2 <= b.page <= o.page + 2]
                if len(samples) >= questions_per_doc:
                    break

                images = [convert_text_to_image(p.image_string) for p in pages]
                name = random.choice(model_names)
                model = select_model(name)

                if check_is_page_multimodal(pages, model):
                    question: str = model.run([instruction, *images]).strip()
                    if check_is_question_multimodal(question, pages, model):
                        answer = model.run([question, *images]).strip()
                        samples.append(
                            MultimodalSample(
                                question=question,
                                answer=answer,
                                evidence_pages=[p.page for p in pages],
                                source=p,
                                annotator=name,
                            )
                        )
                        print(samples[-1].json(indent=2))
                        print(samples[-1].json(), file=f)


"""
python question_generation.py annotate_object_types data/reports/*.pdf
python question_generation.py generate_questions data/docs/*.json --path_out data/questions.json
"""


if __name__ == "__main__":
    Fire()
