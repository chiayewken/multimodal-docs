import base64
import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple

# noinspection PyPackageRequirements
import fitz
import pandas as pd
import requests
from PIL import Image
from fire import Fire
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from reportlab.platypus import (
    Paragraph,
    Image as DocImage,
    SimpleDocTemplate,
    PageBreak,
)
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results


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
    category: str = ""

    def get_image(self) -> Optional[Image.Image]:
        if self.image_string:
            return convert_text_to_image(self.image_string)

    @classmethod
    def from_image(cls, image: Image.Image, **kwargs):
        return cls(image_string=convert_image_to_text(image), **kwargs)


class ObjectDetector(BaseModel, arbitrary_types_allowed=True):
    def run(self, image: Image.Image) -> List[MultimodalObject]:
        raise NotImplementedError()


class YoloDetector(ObjectDetector):
    repo_id: str = "DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet"
    filename: str = "yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt"
    local_dir: str = "data/yolo"
    client: Optional[YOLO] = None

    def load(self):
        if self.client is None:
            if not Path(self.local_dir, self.filename).exists():
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=self.filename,
                    local_dir=self.local_dir,
                )
            self.client = YOLO(Path(self.local_dir, self.filename))

    def save_image(self, image: Image.Image) -> str:
        text = convert_image_to_text(image)
        hash_id = hashlib.md5(text.encode()).hexdigest()
        path = Path(self.local_dir, f"{hash_id}.png")
        image.save(path)
        return str(path)

    @staticmethod
    def extract_subimage(image: Image.Image, box: List[float]) -> Image.Image:
        return image.crop((round(box[0]), round(box[1]), round(box[2]), round(box[3])))

    def run(self, image: Image.Image) -> List[MultimodalObject]:
        self.load()
        path = self.save_image(image)
        results: List[Results] = self.client(source=[path])
        assert len(results) == 1
        objects = []

        for i, label_id in enumerate(results[0].boxes.cls):
            label = results[0].names[label_id.item()]
            score = results[0].boxes.conf[i].item()
            box: List[float] = results[0].boxes.xyxy[i].tolist()
            subimage = self.extract_subimage(image, box)
            objects.append(
                MultimodalObject(
                    image_string=convert_image_to_text(subimage),
                    category=label,
                    score=score,
                )
            )

        return objects


def test_yolo_detector(
    path_image: str = "data/demo_image_paper.png",
    output_dir: str = "outputs/yolo_demo",
):
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    image = Image.open(path_image)
    model = YoloDetector()
    objects = model.run(image)

    for i, o in enumerate(objects):
        subimage = o.get_image()
        path = Path(output_dir, f"{o.category}_{i}.png")
        path.parent.mkdir(exist_ok=True, parents=True)
        subimage.save(path)
        print(path, subimage.size)


class MultimodalPage(BaseModel):
    number: int
    objects: List[MultimodalObject]
    text: str
    image_string: str
    source: str
    score: float = 0.0

    def get_tables_and_figures(self) -> List[MultimodalObject]:
        return [o for o in self.objects if o.category in ["Table", "Picture"]]

    def get_full_image(self) -> Image.Image:
        return convert_text_to_image(self.image_string)

    @classmethod
    def from_text(cls, text: str):
        return MultimodalPage(
            text=text, number=0, objects=[], image_string="", source=""
        )

    @classmethod
    def from_image(cls, image: Image.Image):
        return MultimodalPage(
            image_string=convert_image_to_text(image),
            number=0,
            objects=[],
            text="",
            source="",
        )


class MultimodalDocument(BaseModel):
    pages: List[MultimodalPage]

    def get_page(self, i: int) -> MultimodalPage:
        pages = [p for p in self.pages if p.number == i]
        assert len(pages) == 1
        return pages[0]

    @classmethod
    def load_from_pdf(cls, path: str, dpi: int = 150, detector: ObjectDetector = None):
        # Each page as an image (with optional extracted text)
        doc = fitz.open(path)
        pages = []

        for i, page in enumerate(tqdm(doc.pages(), desc=path)):
            text = page.get_text()
            zoom = dpi / 72  # 72 is the default DPI
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            objects = []
            if detector:
                objects = detector.run(image)
            for o in objects:
                o.page, o.source = i + 1, path

            pages.append(
                MultimodalPage(
                    number=i + 1,
                    objects=objects,
                    text=text,
                    image_string=convert_image_to_text(image),
                    source=path,
                )
            )

        return cls(pages=pages)

    @classmethod
    def load(cls, path: str):
        pages = []
        with open(path) as f:
            for line in f:
                pages.append(MultimodalPage(**json.loads(line)))
        return cls(pages=pages)

    def save(self, path: str):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            for o in self.pages:
                print(o.model_dump_json(), file=f)

    def get_domain(self) -> str:
        filename = Path(self.pages[0].source).name
        if filename.startswith("NYSE"):
            return "Financial<br>Report"
        elif filename[:4].isdigit() and filename[4] == "." and filename[5].isdigit():
            return "Academic<br>Paper"
        else:
            return "Technical<br>Manuals"


class Judgement(BaseModel):
    name: str
    content: str
    score: int


class MultimodalSample(BaseModel):
    question: str
    answer: str
    category: str
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
                print(s.model_dump_json(), file=f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            samples = [MultimodalSample(**json.loads(line)) for line in f]
        return cls(samples=samples)

    def load_documents(self) -> Dict[str, MultimodalDocument]:
        documents = {}
        for s in self.samples:
            if s.source not in documents:
                documents[s.source] = MultimodalDocument.load(s.source)
        return documents


def download_file(url, filename: str = None, overwrite: bool = False):
    if filename is None:
        filename = url.split("/")[-1]
    if Path(filename).exists() and not overwrite:
        print(f"Skipping: {filename}")
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"Downloaded: {filename}")


def download_pdfs(path: str, output_dir: str):
    df = pd.read_csv(path).sample(frac=1, random_state=0)  # Distribute domains
    print(df.shape)
    print(df.head())

    for url in tqdm(df["url"], desc=output_dir):
        filename = Path(output_dir, Path(url).name)
        download_file(url, str(filename), overwrite=False)


def process_documents(*paths: str):
    # Parse the pdfs into images and convert to json format
    detector = YoloDetector()
    for p in tqdm(paths):
        doc = MultimodalDocument.load_from_pdf(p, detector=detector)
        path_out = Path(p).with_suffix(".json")
        path_out.parent.mkdir(parents=True, exist_ok=True)
        doc.save(str(path_out))
        print(dict(path_out=str(path_out), pages=len(doc.pages)))


def save_multimodal_document(
    content: List[Union[str, Image.Image]],
    path: str,
    max_size: int = 512,
    pagesize: Tuple[int, int] = (595, 842),
):
    story = []
    for item in tqdm(content, desc=path):
        if isinstance(item, str):
            if item == "":
                story.append(PageBreak())
            else:
                story.append(Paragraph(item))
        elif isinstance(item, Image.Image):
            ratio = min(max_size / item.width, max_size / item.height)
            width = round(item.width * ratio)
            height = round(item.height * ratio)
            content = io.BytesIO()
            item.save(content, format=item.format)
            story.append(DocImage(content, width=width, height=height))

    Path(path).parent.mkdir(exist_ok=True, parents=True)
    template = SimpleDocTemplate(path, pagesize=pagesize)
    template.build(story)
    print(Path(path).absolute())


"""
python data_loading.py download_pdfs data/train/metadata.csv data/train
python data_loading.py download_pdfs data/test/metadata.csv data/test
python data_loading.py test_yolo_detector
p data_loading.py process_documents data/train/*.pdf
p data_loading.py process_documents data/test/*.pdf
"""


if __name__ == "__main__":
    Fire()
