import base64
import io
import json
from pathlib import Path
from typing import List, Optional

# noinspection PyPackageRequirements
import fitz
import pandas as pd
import requests
from PIL import Image
from fire import Fire
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

    @classmethod
    def load_from_pdf_new(cls, path: str, dpi: int = 150):
        # Each page as an image (with optional extracted text)
        doc = fitz.open(path)
        parts = []

        for i, page in enumerate(tqdm(doc.pages(), desc=path)):
            text = page.get_text()
            zoom = dpi / 72  # 72 is the default DPI
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            parts.append(
                MultimodalObject(
                    id=f"{Path(path).stem}_page_{i + 1}",
                    text=text,
                    page=i + 1,
                    source=path,
                    image_string=convert_image_to_text(image),
                )
            )

        return cls(objects=parts, source=path)

    def as_pages(self) -> List["MultimodalDocument"]:
        num_pages = max(o.page for o in self.objects) + 1
        groups = {i: [] for i in range(num_pages)}
        for o in self.objects:
            groups[o.page].append(o)

        return [MultimodalDocument(objects=groups[i]) for i in sorted(groups)]

    @classmethod
    def load(cls, path: str):
        objects = []
        with open(path) as f:
            for line in f:
                objects.append(MultimodalObject(**json.loads(line)))
        return cls(objects=objects)

    def save(self, path: str):
        with open(path, "w") as f:
            for o in self.objects:
                print(o.json(), file=f)


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


class Judgement(BaseModel):
    name: str
    content: str
    score: int


class MultimodalSample(BaseModel):
    question: str
    answer: str
    evidence_pages: List[int] = []
    raw_output: str = ""
    pred: str = ""
    source: str = ""
    annotator: str = ""
    generator: str = ""
    retrieved_pages: List[int] = []
    judgements: List[Judgement] = []


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


def process_documents(*paths: str, output_dir: str = "data/docs"):
    # Parse the pdfs into images and convert to json format
    for p in tqdm(paths):
        doc = MultimodalDocument.load_from_pdf_new(p)
        path_out = Path(output_dir, Path(p).stem).with_suffix(".json")
        path_out.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(path_out))
        print(dict(path_out=str(path_out), pages=len(doc.objects)))


"""
p data_loading.py test_load_from_excel 
p data_loading.py process_documents data/reports/*.pdf
"""


if __name__ == "__main__":
    Fire()
