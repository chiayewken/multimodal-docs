from pathlib import Path

import pandas as pd
from PIL import Image
from fire import Fire
from tqdm import tqdm

from data_loading import (
    MultimodalDocument,
    MultimodalData,
    MultimodalSample,
    MultimodalObject,
    convert_image_to_text,
)
from modeling import select_model
from retrieval import select_retriever


def load_demo_data(folder: str):
    df = pd.read_excel(Path(folder, "captions.xlsx"))
    mapping = {filename: caption for filename, caption in df.values}
    objects = []

    for path in sorted(Path(folder).glob("*/*.jpg")):
        caption = mapping.get(path.name)
        if caption is not None:
            objects.append(
                MultimodalObject(
                    id=path.name,
                    text=caption,
                    image_string=convert_image_to_text(Image.open(path)),
                )
            )

    for path in sorted(Path(folder).glob("*/*.txt")):
        with open(path) as f:
            text = f.read().strip()
            if not text:
                print(dict(empty_text=path))
                continue
            objects.append(MultimodalObject(id=path.name, text=text))
    doc = MultimodalDocument(objects=objects)
    doc.print()

    samples = []
    df = pd.read_excel(Path(folder, "questions.xlsx"))
    for _, text, _ in df.values:
        question, answer = text.split("Question: ")[1].split("\n\nAnswer: ")
        samples.append(MultimodalSample(question=question, answer=answer, doc=doc))
    return MultimodalData(samples=samples)


def make_demo_data(folder: str) -> MultimodalData:
    objects = []
    for path in sorted(Path(folder).iterdir()):
        page = int(path.stem.split("_")[1])
        if path.suffix == ".txt":
            with open(path) as f:
                objects.append(MultimodalObject(page=page, text=f.read()))
        elif path.suffix == ".png":
            text = " ".join(path.stem.split("_")[2:])
            image = convert_image_to_text(Image.open(path))
            objects.append(MultimodalObject(page=page, text=text, image_string=image))
        else:
            raise ValueError(f"Invalid file: {str(path)}")

    doc = MultimodalDocument(objects=sorted(objects, key=lambda o: o.page))
    doc.print()

    return MultimodalData(
        samples=[
            MultimodalSample(
                question="When did the company complete their initial public offering (IPO)?",
                answer="November 2022",
                doc=doc,
            ),
            MultimodalSample(
                question="What's the cash flow from purchases of property and equipment in 2022?",
                answer="$2,166 thousand",
                doc=doc,
            ),
            MultimodalSample(
                question="In Phase 2 trial of ACR-368, how many patients are still on study?",
                answer="4",
                doc=doc,
            ),
            MultimodalSample(
                question="How much are the Level 1 assets of the company as of December 31 2022 and why are they classified as Level 1 assets?",
                answer="$63,463 thousand. These assets have been valued using quoted market prices in active markets without any valuation adjustment.",
                doc=doc,
            ),
        ]
    )


def main(
    data_dir: str = "data/demo/acrv",
    generator_name: str = "openai",
    retriever_name: str = "clip_text",
    top_k: int = 2,
    output_dir: str = "outputs/demo/acrv",
    **kwargs,
):
    try:
        data = load_demo_data(data_dir)
    except Exception as e:
        print(e)
        print("Using default demo data")
        data = make_demo_data(data_dir)

    generator = select_model(generator_name)
    retriever = select_retriever(retriever_name, top_k=top_k, **kwargs)
    path_out = Path(output_dir, generator_name, retriever_name, f"top_k_{top_k}.jsonl")

    for sample in tqdm(data.samples, desc=str(path_out)):
        query = MultimodalObject(text=sample.question)
        sample.prompt = retriever.run(query, doc=sample.doc)
        sample.prompt.objects.insert(0, query)

        # Avoid "no image in input" error
        if all(x.image_string == "" for x in sample.prompt.objects):
            sample.prompt.objects.append(
                [x for x in sample.doc.objects if x.image_string][0]
            )

        sample.raw_output = generator.run(sample.prompt)
        sample.print()

    data.save(str(path_out))


"""
p demo.py main --generator_name openai_vision
p demo.py main --generator_name openai_vision --data_dir data/demo/NASDAQ_AMLX_2022 --top_k 10 --output_dir outputs/demo/amlx
"""


if __name__ == "__main__":
    Fire()
