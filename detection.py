import os
import random
import sys
from pathlib import Path
from typing import List

from PIL import Image
from fire import Fire
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.engine.results import Results

from data_loading import MultimodalDocument, MultimodalObject, convert_text_to_image


def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    return image.resize((new_width, new_height), resample=Image.LANCZOS)


def save_detection_data(*paths: str, output_dir: str, num: int, image_size: int = 640):
    # Parse the pdfs into images and convert to json format
    objects = []
    for p in tqdm(paths):
        doc = MultimodalDocument.load_from_pdf(p)
        objects.extend(doc.objects)

    random.seed(0)
    o: MultimodalObject
    for o in tqdm(random.sample(objects, k=num), desc=output_dir):
        name = f"page_{o.page}_{Path(o.source).name}"
        path_out = Path(output_dir, name).with_suffix(".jpg")
        path_out.parent.mkdir(exist_ok=True, parents=True)
        image = resize_image(convert_text_to_image(o.image_string), image_size)
        image.save(path_out)
        print(path_out)


def apply_patch(file_path, old_content, new_content):
    with open(file_path, "r") as file:
        content = file.read()
        if old_content in content:
            print(dict(old=old_content, new=new_content))
            content = content.replace(old_content, new_content)
    with open(file_path, "w") as file:
        file.write(content)


def fix_labeler_source():
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        print("Error: CONDA_PREFIX environment variable not found.")
        sys.exit(1)

    canvas_path = Path(conda_prefix, "lib/python3.10/site-packages/libs/canvas.py")

    apply_patch(
        canvas_path,
        "p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)",
        "p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))",
    )

    apply_patch(
        canvas_path,
        "p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())",
        "p.drawLine( int(self.prev_point.x()), 0, int(self.prev_point.x()), int(self.pixmap.height()))",
    )

    apply_patch(
        canvas_path,
        "p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())",
        "p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()), int(self.prev_point.y()))",
    )

    label_path = Path(conda_prefix, "lib/python3.10/site-packages/labelImg/labelImg.py")

    apply_patch(
        label_path,
        "bar.setValue(bar.value() + bar.singleStep() * units)",
        "bar.setValue(int(bar.value() + bar.singleStep() * units))",
    )

    print("Patches applied successfully.")


def test_yolo(
    repo: str = "DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet",
    filename: str = "yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt",
    folder: str = "data",
    path_image: str = "data/demo_image_paper.png",
):
    hf_hub_download(repo_id=repo, filename=filename, local_dir=folder)
    model = YOLO(Path(folder, filename))

    results: List[Results] = model(
        source=[path_image],
        save=True,
        show_labels=True,
        show_conf=True,
        show_boxes=True,
    )

    print(results)
    breakpoint()


"""
python detection.py fix_labeler_source
python detection.py save_detection_data data/train/*.pdf --output_dir data/detect_train --num 500
labelImg data/detect_train
"""


if __name__ == "__main__":
    Fire()
