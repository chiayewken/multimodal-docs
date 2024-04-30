import base64
import io
import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Optional

import fitz
import pandas as pd
import requests
from PIL import Image
from fire import Fire
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.readers.file import PDFReader
from pydantic import BaseModel
from tqdm import tqdm

from reading import get_doc_images


def convert_image_to_text(image: Image) -> str:
    # This is also how OpenAI encodes images: https://platform.openai.com/docs/guides/vision
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        data = output.getvalue()
    return base64.b64encode(data).decode("utf-8")


def convert_text_to_image(text: str) -> Image:
    data = base64.b64decode(text.encode("utf-8"))
    return Image.open(io.BytesIO(data))


def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    raise ValueError(f"Failed to retrieve image. Status code: {response.status_code}")


class MultimodalObject(BaseModel):
    id: str = ""
    page: int = 0
    text: str = ""
    image_string: str = ""
    snippet: str = ""
    score: float = 0.0
    source: str = ""

    def get_image(self) -> Optional[Image.Image]:
        if self.image_string:
            return convert_text_to_image(self.image_string)

    @classmethod
    def from_image(cls, image: Image.Image, **kwargs):
        return cls(image_string=convert_image_to_text(image), **kwargs)


class MultimodalDocument(BaseModel):
    objects: List[MultimodalObject]
    source: str = ""

    def print(self):
        for x in self.objects:
            x = x.copy(deep=True)
            if len(x.text) > 80:
                x.text = x.text[:80] + "..."
            if x.image_string:
                x.image_string = x.image_string[:20] + "..."
            print(x.json(indent=2))

    @classmethod
    def load_from_folder(cls, folder: str, folder_pdf: str = ""):
        # Folder has jpg image files and a captions.xlsx
        path = sorted(Path(folder).glob("**/*.xlsx"))[0]
        df = pd.read_excel(path, engine="calamine")
        image_map = {x.stem: x for x in Path(folder).glob("**/*.jpg")}
        assert image_map

        parts = []
        for filename, caption in tqdm(df.values[:, :2], desc=folder):
            image_path = image_map.get(Path(filename).stem)
            if image_path is None:
                print(dict(missing_image=Path(filename).stem))
                continue
            image = Image.open(image_path)
            if image.mode == "CMYK":
                image = image.convert("RGB")
                print(dict(special_image_cmyk=image_path))

            numbers = [n for n in Path(filename).stem.split("_")]
            page = numbers[-2] if numbers[-2].isdigit() else numbers[-1]  # fig_16_3.jpg
            parts.append(
                MultimodalObject.from_image(
                    image, text=caption.strip(), page=page, source=str(image_path)
                )
            )

        if folder_pdf:
            # Extract text from PDF
            path = Path(folder_pdf, Path(folder).name).with_suffix(".pdf")
            assert path.exists(), f"PDF not found: {path}"
            reader = PDFReader()
            data = reader.load_data(path)
            for x in data:
                text = MultimodalObject(
                    text=x.text, page=x.metadata["page_label"], source=str(path)
                )
                parts.insert(0, text)  # Prioritize text parts before sorting

        if not parts:
            raise ValueError(f"No multimodal parts found in {folder}")
        return cls(objects=sorted(parts, key=lambda p: p.page), source=folder)

    @classmethod
    def load_from_pdf(cls, path: str):
        doc = fitz.open(path)
        image_map = get_doc_images(doc)
        parts = []

        for i, page in enumerate(doc.pages()):
            text = page.get_text()
            if text.strip():
                parts.append(MultimodalObject(text=text, page=i + 1, source=path))
            for image in image_map.get(i, []):
                parts.append(
                    MultimodalObject.from_image(image, page=i + 1, source=path)
                )

        return cls(objects=parts, source=path)

    def as_pages(self) -> List["MultimodalDocument"]:
        num_pages = max(o.page for o in self.objects) + 1
        groups = {i: [] for i in range(num_pages)}
        for o in self.objects:
            groups[o.page].append(o)

        return [
            MultimodalDocument(objects=groups[i], source=self.source)
            for i in sorted(groups)
        ]


def test_load_from_folder(
    data_dir: str = "raw_data/annotation_first_step",
    folder_pdf: str = "raw_data/annual_reports_2022_selected",
):
    for folder in sorted(Path(data_dir).iterdir()):
        if folder.name.startswith("."):
            continue

        try:
            doc = MultimodalDocument.load_from_folder(str(folder), folder_pdf)
            print(doc.objects[-1].json(indent=2, exclude={"image_string"}))
            print(
                dict(
                    folder=folder,
                    num_images=sum(o.image_string != "" for o in doc.objects),
                    num_texts=sum(o.image_string == "" for o in doc.objects),
                )
            )

        except Exception as e:
            print(e)
            print(folder, folder_pdf)


class MultimodalSample(BaseModel):
    question: str
    answer: str
    doc: MultimodalDocument
    evidence: MultimodalDocument = MultimodalDocument(objects=[])
    prompt: MultimodalDocument = MultimodalDocument(objects=[])
    raw_output: str = ""
    pred: str = ""
    source: str = ""

    def print(self):
        self.doc.print()
        self.prompt.print()
        print(self.json(indent=2, exclude={"prompt", "doc"}))


class MultimodalData(BaseModel):
    samples: List[MultimodalSample]

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for s in self.samples:
                print(s.json(), file=f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            samples = [MultimodalSample(**json.loads(line)) for line in f]
        return cls(samples=samples)

    @classmethod
    def load_from_excel(
        cls,
        path: str,
        data_dir: str = "raw_data/annotation_first_step",
        pdf_dir: str = "raw_data/annual_reports_2022_selected",
    ):
        df = pd.read_excel(path)
        pd.set_option("display.max_columns", None)

        samples = []
        for name, group in df.groupby("pdf_file"):
            assert isinstance(name, str)
            doc = MultimodalDocument.load_from_folder(
                folder=str(Path(data_dir, name)), folder_pdf=pdf_dir
            )
            evidence_map = {Path(o.source).name: [o] for o in doc.objects}

            for raw in group.to_dict(orient="records"):
                if raw["question_correctness"] + raw["answer_correctness"] == "YY":
                    question, answer = raw["QA"].split("\n\nAnswer: ")
                    sample = MultimodalSample(
                        question=question.split("Question: ")[-1].strip(),
                        answer=answer.strip(),
                        evidence=MultimodalDocument(
                            objects=evidence_map.get(raw["file_segmented"], [])
                        ),
                        doc=doc,
                        source=path,
                    )
                    samples.append(sample)

        print(dict(path=path, samples=len(samples)))
        return cls(samples=samples)

    @classmethod
    def load_from_excel_and_pdf(
        cls, path: str, pdf_dir: str = "raw_data/annual_reports_2022_selected"
    ) -> "MultimodalData":
        df = pd.read_excel(path)
        pd.set_option("display.max_columns", None)

        samples = []
        for name, group in df.groupby("pdf_file"):
            assert isinstance(name, str)
            doc = MultimodalDocument.load_from_pdf(
                path=str(Path(pdf_dir, name).with_suffix(".pdf"))
            )

            for raw in group.to_dict(orient="records"):
                numbers = [n for n in Path(raw["file_segmented"]).stem.split("_")]
                page = int(numbers[-2] if numbers[-2].isdigit() else numbers[-1])
                evidence = MultimodalObject(source=raw["file_segmented"], page=page)
                if raw["question_correctness"] + raw["answer_correctness"] == "YY":
                    question, answer = raw["QA"].split("\n\nAnswer: ")
                    sample = MultimodalSample(
                        question=question.split("Question: ")[-1].strip(),
                        answer=answer.strip(),
                        evidence=MultimodalDocument(objects=[evidence]),
                        doc=doc,
                        source=path,
                    )
                    samples.append(sample)

        print(dict(path=path, samples=len(samples)))
        return cls(samples=samples)

    def analyze(self):
        random.seed(0)
        sample: MultimodalSample
        for sample in random.sample(self.samples, k=5):
            e = sample.evidence.objects[0]
            lst = [(o.text, o.source) for o in sample.doc.objects if o.page == e.page]

            raw = dict(
                page_objects=lst,
                question=sample.question,
                answer=sample.answer,
                evidence=e.text,
                source=sample.doc.source,
                page=e.page,
            )
            print(json.dumps(raw, indent=2))

        info = dict(
            samples=len(self.samples),
            sources=str(Counter(s.doc.source for s in self.samples)),
            with_evidence=sum(len(s.evidence.objects) > 0 for s in self.samples),
        )
        print(json.dumps(info, indent=2))


def test_load_from_excel(path: str = "data/财报标注-0416.xlsx"):
    data = MultimodalData.load_from_excel(path)
    data.analyze()
    breakpoint()


def test_read_pdf(
    path: str = "raw_data/annual_reports_2022_selected/NASDAQ_ACRV_2022.pdf",
):
    reader = SimpleDirectoryReader(input_files=[path])
    data: List[Document] = reader.load_data()
    print(len(data))

    reader2 = PDFReader()
    data2 = reader2.load_data(Path(path))
    assert len(data) == len(data2)
    for a, b in zip(data, data2):
        assert a.text == b.text
    breakpoint()


"""
p data_loading.py test_load_from_excel 
"""


if __name__ == "__main__":
    Fire()
