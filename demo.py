from pathlib import Path

from PIL import Image
from fire import Fire

from data_loading import (
    MultimodalDocument,
    MultimodalData,
    MultimodalSample,
    MultimodalObject,
    convert_image_to_text,
)
from modeling import select_model
from retrieval import select_retriever


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
    generator_name: str = "gemini_vision",
    retriever_name: str = "clip_text",
    top_k: int = 2,
    output_dir: str = "outputs/demo/acrv",
    **kwargs,
):
    data = make_demo_data(data_dir)
    generator = select_model(generator_name)
    retriever = select_retriever(retriever_name, **kwargs)

    for sample in data.samples:
        query = MultimodalObject(text=sample.question)
        sample.prompt = retriever.run(query, doc=sample.doc).get_top_objects(k=top_k)
        sample.prompt.objects.insert(0, query)

        # Avoid "no image in input" error
        if all(x.image_string == "" for x in sample.prompt.objects):
            sample.prompt.objects.append(
                [x for x in sample.doc.objects if x.image_string][0]
            )

        sample.raw_output = generator.run(sample.prompt)
        sample.print()

    path_out = Path(output_dir, generator_name, retriever_name, f"top_k_{top_k}.jsonl")
    data.save(str(path_out))


if __name__ == "__main__":
    Fire()
