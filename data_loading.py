import base64
import hashlib
import io
import json
import random
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
        for s in tqdm(self.samples, desc="load_documents"):
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


def get_domain(url: str) -> str:
    filename = Path(url).name
    page = MultimodalPage(
        number=0, objects=[], text="", image_string="", source=filename
    )
    doc = MultimodalDocument(pages=[page])
    return doc.get_domain()


def download_pdfs(path: str, output_dir: str):
    df = pd.read_csv(path).sample(frac=1, random_state=0)  # Distribute domains
    print(df.shape)
    print(df.head())

    # Check all urls are unique
    assert df["url"].nunique() == df.shape[0], df[
        df.duplicated(subset=["url"], keep=False)
    ].values
    df["domain"] = df["url"].apply(get_domain)
    print(df["domain"].value_counts())

    for url in tqdm(df["url"], desc=output_dir):
        filename = Path(output_dir, Path(url).name)
        suffix = "" if filename.suffix == ".pdf" else ".pdf"
        download_file(url, str(filename) + suffix, overwrite=False)


def process_documents(*paths: str, exclude: List[str] = (), skip_exist: bool = False):
    # Parse the pdfs into images and convert to json format
    detector = YoloDetector()
    lst = []
    for p in paths:
        if any(Path(p).name.startswith(str(prefix)) for prefix in exclude):
            continue
        if skip_exist and Path(p).with_suffix(".json").exists():
            continue
        lst.append(p)

    for p in tqdm(lst):
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


def sample_questions(path_in: str, path_out: str, num: int, seed: int = 0):
    random.seed(seed)
    data = MultimodalData.load(path_in)
    print(dict(path=path_in, samples=len(data.samples)))
    data.samples = random.sample(data.samples, num)
    data.save(path_out)
    print(dict(path=path_out, samples=len(data.samples)))


def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    # Same as modeling.py resize_image
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    if width > height:
        new_width = max_size
        new_height = round(height * max_size / width)
    else:
        new_height = max_size
        new_width = round(width * max_size / height)
    return image.resize((new_width, new_height), Image.LANCZOS)


def make_qwen_train_inputs(
    sample: MultimodalSample,
    top_k: int,
    use_gold_page_only: bool = False,
    documents: dict = None,
) -> Tuple[str, List[Image.Image]]:
    # Adapted from evaluation.py generate_answers
    if documents is None:
        doc = MultimodalDocument.load(sample.source)
    else:
        if sample.source not in documents:
            documents[sample.source] = MultimodalDocument.load(sample.source)
        doc = documents[sample.source]

    assert sample.retrieved_pages
    sample.retrieved_pages = sorted(sample.retrieved_pages[:top_k])

    context = []
    for p in doc.pages:
        sample = sample.copy(deep=True)
        if use_gold_page_only:
            sample.retrieved_pages = sample.evidence_pages
            assert len(sample.evidence_pages) == 1
        if p.number in sample.retrieved_pages:
            if p.text:
                context.append(p.text)
            context.extend(o.get_image() for o in p.get_tables_and_figures())

    inputs = [
        "Context:",
        *context,
        f"Answer the following question in 200 words or less: {sample.question}",
    ]
    text = "\n\n".join([x for x in inputs if isinstance(x, str)])
    for x in inputs:
        if isinstance(x, Image.Image):
            text = "<image>" + text

    images = [resize_image(x, 768) for x in inputs if isinstance(x, Image.Image)]
    assert text.count("<image>") == len(images)
    return text, images


def save_image(image: Image.Image, folder: str) -> str:
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    path = Path(folder, f"{image_hash}.png")
    path.parent.mkdir(exist_ok=True, parents=True)
    if not path.exists():
        image.save(path)
    return str(path)


def make_qwen_data(
    *paths: str,
    name: str,
    path_info: str = "data/dataset_info.json",
    image_dir: str = "data/qwen_images",
    limit: int = 0,
):
    print(locals())
    path_out = Path(Path(path_info).parent, name).with_suffix(".json")
    with open(path_info) as f:
        info = json.load(f)
    with open(path_info, "w") as f:
        info[name] = json.loads(json.dumps(info["mllm_demo"]))  # Copy
        info[name]["file_name"] = str(path_out.name)
        json.dump(info, f, indent=2)

    data = []
    for p in paths:
        for sample in tqdm(MultimodalData.load(p).samples):
            if 0 < limit <= len(data):
                continue
            text, images = make_qwen_train_inputs(sample, top_k=5)
            assert sample.answer.strip() != ""
            messages = [
                dict(role="user", content=text),
                dict(role="assistant", content=sample.answer),
            ]

            image_paths = []
            for x in images:
                image_paths.append(save_image(x, image_dir).split("/", maxsplit=1)[1])
            data.append(dict(messages=messages, images=image_paths))

    with open(path_out, "w") as f:
        json.dump(data, f, indent=2)
    print(dict(path_out=path_out, samples=len(data)))


def load_excel_annotation(*paths: str, path_out: str):
    questions = []
    for p in paths:
        df = pd.read_excel(p)
        df = df.dropna(subset=["valid"])
        print(dict(p=p, df=df.shape, valid=df["valid"].sum()))
        questions.extend(df[df["valid"] == 1]["question"].tolist())

    print(dict(questions=len(questions), unique=len(set(questions))))
    with open(path_out, "w") as f:
        json.dump(questions, f, indent=2)


def load_valid_questions(path: str) -> List[str]:
    with open(path) as f:
        return json.load(f)


def make_swift_qwen_data(
    *paths: str,
    path_out: str,
    image_dir: str = "data/qwen_images",
    limit: int = 0,
    is_test: bool = False,
    use_gold_page_only: bool = False,
    use_pred_as_reject_response: bool = False,
):
    print(locals())
    total = 0
    Path(path_out).parent.mkdir(exist_ok=True, parents=True)
    all_samples = [s for p in paths for s in MultimodalData.load(p).samples]
    documents = {}

    with open(path_out, "w") as f:
        for sample in tqdm(all_samples, desc=path_out):
            if not (0 < limit <= total):
                # {"query": "<image>55555", "response": "66666", "images": ["image_path"]}
                assert sample.answer.strip() != "" or is_test
                text, images = make_qwen_train_inputs(
                    sample,
                    top_k=5,
                    use_gold_page_only=use_gold_page_only,
                    documents=documents,
                )

                image_paths = [save_image(x, image_dir) for x in images]
                info = dict(query=text, response=sample.answer, images=image_paths)
                if use_pred_as_reject_response:
                    assert sample.pred.strip() != ""
                    info["rejected_response"] = sample.pred

                print(json.dumps(info), file=f)
                # print(info)
                total += 1


def get_latest_infer_file(path: str) -> str:
    # eg output/qwen2-vl-7b-instruct/v12-20241001-202206/checkpoint-623 -> .../checkpoint-623/infer_result/20241002-013038.jsonl
    assert Path(path).name.startswith("checkpoint")
    output = ""
    latest_date = 0
    latest_time = 0

    for p in Path(path).glob("infer_result/*.jsonl"):
        date, time = [int(x) for x in p.stem.split("-")]
        if date > latest_date or (date == latest_date and time > latest_time):
            output = str(p)
            latest_date, latest_time = date, time

    print(dict(latest_infer_file=output))
    return output


def read_swift_qwen_preds(path: str, path_questions: str, path_out: str):
    if not path.endswith(".jsonl"):
        path = get_latest_infer_file(path)
    with open(path) as f:
        raw = [json.loads(line) for line in f]
    data = MultimodalData.load(path_questions)
    assert len(raw) == len(data.samples)
    for i, sample in enumerate(data.samples):
        assert sample.question in raw[i]["query"]
        sample.pred = raw[i]["response"]
    data.save(path_out)
    print(path_out)


"""
python data_loading.py download_pdfs data/train/metadata.csv data/train
python data_loading.py download_pdfs data/test/metadata.csv data/test
python data_loading.py test_yolo_detector
p data_loading.py process_documents data/train/*.pdf --skip_exist
p data_loading.py process_documents data/test/*.pdf
p data_loading.py process_documents data/test/NYSE*.pdf
p data_loading.py process_documents data/test/24*.pdf
p data_loading.py process_documents data/test/*.pdf --exclude "NYSE,24"

################################################################################
Training data

python data_loading.py make_swift_qwen_data outputs/retrieve/train*/colpali.json --path_out data/swift/train_10k.jsonl
python data_loading.py make_swift_qwen_data outputs/retrieve/train*/colpali.json --path_out data/swift/train_18k.jsonl

python data_loading.py make_swift_qwen_data outputs/retrieve/test/colpali.json --path_out data/swift/test.jsonl --is_test
python data_loading.py make_swift_qwen_data outputs/retrieve/test/colpali_sample_100.json --path_out data/swift/test_sample_100.jsonl --is_test 

#### 10k training data with qwen2-vl-7b-instruct

swift sft \
--rescale_image 240000 \
--max_length 6144 \
--lora_rank 64 \
--model_type qwen2-vl-7b-instruct \
--sft_type lora \
--dataset data/swift/train_10k.jsonl

# Manual infer and test
swift infer --ckpt_dir output/qwen2-vl-7b-instruct/v12-20241001-202206/checkpoint-623 --val_dataset data/swift/test.jsonl
python data_loading.py read_swift_qwen_preds output/qwen2-vl-7b-instruct/v12-20241001-202206/checkpoint-623/infer_result/20241002-013038.jsonl outputs/retrieve/test/colpali.json outputs/swift_qwen_10k/colpali/top_k=5.json

# Automatic infer and test
bash scripts/eval_swift.sh outputs/retrieve/test/colpali_sample_100.json \
outputs_swift/train-qwen \
output/qwen2-vl-7b-instruct/v12-20241001-202206/checkpoint-623
[35:50<00:00, 21.50s/it, score=3.99]

# 10k -> 18k training data improves performance slightly

swift sft \
--rescale_image 240000 \
--max_length 6144 \
--lora_rank 64 \
--model_type qwen2-vl-7b-instruct \
--sft_type lora \
--dataset data/swift/train_18k.jsonl

bash scripts/eval_swift.sh outputs/retrieve/test/colpali_sample_100.json \
outputs_swift/train-qwen-18k \
output/qwen2-vl-7b-instruct/v37-20241018-235007/checkpoint-1125
[22:05<00:00, 13.26s/it, score=4.02]

################################################################################
# Ablation: Training single-page, testing multi-page
python data_loading.py make_swift_qwen_data outputs/retrieve/train*/colpali.json --path_out data/swift/train_single_page.jsonl --use_gold_page_only
python data_loading.py make_swift_qwen_data outputs/retrieve/test/colpali.json --path_out data/swift/test_single_page.jsonl --is_test
python data_loading.py make_swift_qwen_data outputs/retrieve/test/colpali_sample_100.json --path_out data/swift/test_sample_100_single_page.jsonl --is_test

swift sft \
--eval_steps 9999 \
--save_steps 100 \
--rescale_image 240000 \
--max_length 6144 \
--lora_rank 64 \
--model_type qwen2-vl-7b-instruct \
--sft_type lora \
--dataset data/swift/train_single_page.jsonl

swift infer --ckpt_dir output/qwen2-vl-7b-instruct/v17-20241002-173257/checkpoint-623 --val_dataset data/swift/test_sample_100_single_page.jsonl
python data_loading.py read_swift_qwen_preds output/qwen2-vl-7b-instruct/v17-20241002-173257/checkpoint-623/infer_result/20241011-162621.jsonl outputs/retrieve/test/colpali_sample_100.json outputs/swift_qwen_10k_single_page/colpali/top_k=5.json
python evaluation.py run_multi_judge outputs/swift_qwen_10k_single_page/colpali/top_k=5.json
# swift_qwen_10k_single_page      3.96     3.99     3.95  4.20    4.02   3.65  3.96

bash scripts/eval_swift.sh outputs/retrieve/test/colpali_sample_100.json \
outputs_swift/train_qwen_single_page \
output/qwen2-vl-7b-instruct/v17-20241002-173257/checkpoint-623
[36:40<00:00, 22.01s/it, score=3.94]

################################################################################
# Three epochs instead of one
swift sft \
--num_train_epochs 3 \
--rescale_image 240000 \
--max_length 6144 \
--lora_rank 64 \
--model_type qwen2-vl-7b-instruct \
--sft_type lora \
--dataset data/swift/train_10k.jsonl

bash scripts/eval_swift.sh outputs/retrieve/test/colpali_sample_100.json \
outputs_swift/train_qwen_3_epochs \
output/qwen2-vl-7b-instruct/v19-20241015-141048/checkpoint-1869
[25:17<00:00, 15.17s/it, score=3.98]

################################################################################
# Full-parameter instead of lora
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 swift sft \
--sft_type full \
--freeze_vit true \
--rescale_image 240000 \
--max_length 6144 \
--model_type qwen2-vl-7b-instruct \
--dataset data/swift/train_10k.jsonl

sleep 1800 && bash scripts/eval_swift.sh outputs/retrieve/test/colpali_sample_100.json \
outputs_swift/train_qwen_full_params \
output/qwen2-vl-7b-instruct/v20-20241015-142637/checkpoint-623
[29:30<00:00, 17.70s/it, score=3.94]

bash scripts/eval_swift.sh outputs/retrieve/test/colpali_sample_100.json \
outputs_swift/train-qwen-4gpu-maxlen-8192-lora-target-all \
output/qwen2-vl-7b-instruct/v33-20241017-180302/checkpoint-600
[27:14<00:00, 16.34s/it, score=3.9]

################################################################################
# Contrastive (Needs at least 2x A800)

python data_loading.py make_swift_qwen_data outputs/qwen/colpali/top_k=5_remove_gold_train*.json --path_out data/swift/train_10k_with_reject.jsonl --use_pred_as_reject_response

CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 swift rlhf \
--rlhf_type orpo \
--beta 0.1 \
--rescale_image 240000 \
--max_length 6144 \
--lora_rank 64 \
--model_type qwen2-vl-7b-instruct \
--sft_type lora \
--dataset data/swift/train_10k_with_reject.jsonl

python data_loading.py make_swift_qwen_data outputs/qwen/colpali/top_k=5_remove_gold_train*.json --path_out data/swift/train_10k_with_reject_single_page.jsonl --use_pred_as_reject_response --use_gold_page_only

swift rlhf \
--rlhf_type orpo \
--beta 0.1 \
--rescale_image 240000 \
--max_length 6144 \
--lora_rank 64 \
--model_type qwen2-vl-7b-instruct \
--sft_type lora \
--dataset data/swift/train_10k_with_reject_single_page.jsonl

bash scripts/eval_swift.sh outputs/retrieve/test/colpali_sample_100.json \
outputs_swift/train-qwen-with-reject-single-page \
output/qwen2-vl-7b-instruct/v38-20241019-000933/checkpoint-623
[21:08<00:00, 12.69s/it, score=3.98]

bash scripts/eval_swift.sh outputs/retrieve/test/colpali_sample_100.json \
outputs_swift/train-qwen-use-dora \
output/qwen2-vl-7b-instruct/v36-20241018-200841/checkpoint-623
[20:19<00:00, 12.20s/it, score=3.94]

################################################################################
# Pixtral cuda error
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
--rescale_image 240000 \
--max_length 6144 \
--lora_rank 64 \
--model_type pixtral-12b \
--sft_type lora \
--dataset data/swift/train_10k.jsonl

################################################################################
Annotation data

p data_loading.py load_excel_annotation \
data/annotation/academic_cq.xlsx \
data/annotation/product_cq.xlsx \
data/annotation/product_mj.xlsx \
data/annotation/finance_ma.xlsx \
data/annotation/product_ma.xlsx \
--path_out data/annotation/valid_questions.json
"""


if __name__ == "__main__":
    Fire()
