import io
import random
import re
from pathlib import Path

# noinspection PyPackageRequirements
import fitz  # imports the pymupdf library
import pandas as pd
from fire import Fire
from reportlab.platypus import (
    SimpleDocTemplate,
    Image as DocImage,
    Paragraph,
    PageBreak,
)
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from data_loading import (
    load_image_from_url,
    convert_text_to_image,
    MultimodalSample,
    MultimodalData,
    MultimodalDocument,
    MultimodalObject,
)
from modeling import OpenAIMiniModel, select_model
from reading import get_doc_images


def test_clip(
    image_url: str = "https://github.com/UKPLab/sentence-transformers/raw/master/examples/applications/image-search/two_dogs_in_snow.jpg",
):
    model = SentenceTransformer("clip-ViT-L-14")
    image = load_image_from_url(image_url)
    # noinspection PyTypeChecker
    query = model.encode(image)
    texts = [
        "Two dogs in the snow",
        "A cat on a table",
        "A picture of London at night",
        "A black dog and a white dog in the snow",
        image,
    ]
    data = model.encode(texts)

    # Compute cosine similarities
    cos_scores = util.cos_sim(query, data).squeeze().tolist()
    for i, score in enumerate(cos_scores):
        print(dict(text=texts[i], score=cos_scores[i]))


def load_doc_image(text: str, size: int) -> DocImage:
    image = convert_text_to_image(text)
    content = io.BytesIO()
    image.save(content, format=image.format)
    height = size * image.height // image.width
    return DocImage(content, width=size, height=height)


def show_preds(path: str, path_out: str, image_size=256):
    Path(path_out).parent.mkdir(exist_ok=True, parents=True)
    doc = SimpleDocTemplate(str(path_out), pagesize=(8.5 * 72, 25.5 * 72))
    story = []
    retrieve_success = []

    sample: MultimodalSample
    for i, sample in enumerate(MultimodalData.load(path).samples):
        sample.print()
        story.append(Paragraph(f"<b>Prompt for {sample.doc.source}</b>:"))
        for x in sample.prompt.objects:
            story.append(Paragraph(f"<b>Page {x.page}</b>:"))
            if x.text:
                story.append(Paragraph(x.text))
            if x.image_string:
                story.append(load_doc_image(x.image_string, image_size))

        story.append(Paragraph(f"<b>Gold Answer</b>: {sample.answer}"))
        story.append(Paragraph(f"<b>Raw Output</b>: {sample.raw_output}"))
        story.append(Paragraph(f"<b>Evidence</b>: {sample.evidence.objects[0].page}"))
        retrieve_success.append(
            sample.evidence.objects[0].page in [o.page for o in sample.prompt.objects]
        )
        story.append(Paragraph(f"<b>Retrieve Success</b>: {retrieve_success[-1]}"))
        story.append(PageBreak())

    doc.build(story)
    print(Path(path_out).absolute())
    print(dict(retrieve_success=sum(retrieve_success) / len(retrieve_success)))


def check_empty_texts(*paths: str):
    for p in paths:
        with open(p) as f:
            if f.read().strip() == "":
                print(dict(empty=p))


def test_pdf_reader(path: str, path_out: str = "example.pdf"):
    template = SimpleDocTemplate(str(path_out))
    story = []

    # PyMuPDF is better than pypdf as pypdf sometimes joins words together
    doc = fitz.open(path)  # open a document
    image_map = get_doc_images(doc)

    for i, page in enumerate(doc.pages()):  # iterate the document pages
        text = page.get_text()  # get plain text encoded as UTF-8
        print(text)

        story.append(Paragraph(f"<b>Page {i}</b>"))
        story.append(Paragraph(f"<b>Text</b>"))
        story.append(Paragraph(text))

        for image in image_map.get(i, []):
            story.append(Paragraph(f"<b>Image</b>"))
            content = io.BytesIO()
            image.save(content, format=image.format)
            story.append(DocImage(content, width=256, height=256))

    template.build(story)
    print(Path(path_out).absolute())


def test_load_from_pdf(
    path: str = "raw_data/annual_reports_2022_selected/NASDAQ_VERV_2022.pdf",
    path_out: str = "example.pdf",
):
    template = SimpleDocTemplate(str(path_out))
    story = []
    doc = MultimodalDocument.load_from_pdf(path)

    for i, page in enumerate(doc.as_pages()):  # iterate the document pages
        story.append(Paragraph(f"<b>Page {i}</b>"))
        for o in page.objects:
            if o.text:
                story.append(Paragraph(f"<b>Text</b>"))
                story.append(Paragraph(o.text))
            if o.image_string:
                story.append(load_doc_image(o.image_string, 256))

    template.build(story)
    print(Path(path_out).absolute())


def read_index(path: str = "data/nyse.txt", seed: int = 0):
    # https://www.annualreports.com/Companies?exch=1
    # https://uk.marketscreener.com/quote/index/MSCI-WORLD-107361487/components/
    records = []
    with open(path) as f:
        content = f.read().replace("\nRequest", "Request\n")
        for i, chunk in enumerate(content.split("\n\n")):
            if i == 0 or "Request" in chunk:
                continue  # Skip header
            if len(chunk.split("\n")) != 3:
                continue
            a, b, c = chunk.split("\n")
            records.append(dict(name=a.strip(), industry=b.strip(), sector=c.strip()))

    df = pd.DataFrame(records)
    print(df.shape)
    pd.set_option("display.max_rows", None)
    print(df.sample(100, random_state=seed))


def test_read_pdf_new(path: str = "data/reports/NYSE_HI_2023.pdf"):
    doc = MultimodalDocument.load_from_pdf_new(path)
    path_out = Path("renders", Path(path).name)
    Path(path_out).parent.mkdir(exist_ok=True, parents=True)

    story = []
    random.seed(0)
    objects = sorted(random.sample(doc.objects, 10), key=lambda x: x.page)
    for x in objects:
        story.append(Paragraph(f"<b>Page {x.id}</b>:"))
        image = convert_text_to_image(x.image_string)
        story.append(Paragraph(f"<b>Image {image.size}</b>:"))
        story.append(load_doc_image(x.image_string, size=256))
        story.append(Paragraph(f"<b>Text</b>:"))
        story.append(Paragraph(x.text))
        story.append(PageBreak())

    template = SimpleDocTemplate(str(path_out))
    template.build(story)
    print(Path(path_out).absolute())


def test_scores(path: str):
    data = MultimodalData.load(path)
    scores = [j.score for s in data.samples for j in s.judgements]
    print(sum(scores) / len(scores))

    print("Scores by each judge")
    judges = [j.name for s in data.samples for j in s.judgements]
    for name in sorted(set(judges)):
        scores = [j.score for s in data.samples for j in s.judgements if j.name == name]
        empty = sum(s == -1 for s in scores) / len(scores)
        print(dict(name=name, score=sum(scores) / len(scores), empty=empty))


def test_doc_content(
    *paths: str, pages_per_doc: int = 1, path_out: str = "renders/tags.pdf"
):
    random.seed(0)
    model = select_model("openai_mini")
    instruction = "Identify all the content types in the document. Output one or more of the following tags: <text>, <figure>, <table>."
    tags = []
    contents = []

    for p in tqdm(paths):
        doc = MultimodalDocument.load(p)
        o: MultimodalObject
        for o in random.sample(doc.objects, k=pages_per_doc):
            image = o.get_image()
            text = model.run([instruction, image])
            pattern = r"<(\w+)>"
            tags.append(re.findall(pattern, text))
            print(tags[-1])

            contents.append(load_doc_image(o.image_string, 256))
            contents.append(Paragraph(o.source))
            contents.append(Paragraph(str(tags[-1])))
            contents.append(PageBreak())

    template = SimpleDocTemplate(str(path_out), pagesize=(8.5 * 72, 25.5 * 72))
    template.build(contents)
    print(Path(path_out).absolute())


def test_doc_parsing(
    *paths: str, pages_per_doc: int = 1, path_out: str = "renders/parse.pdf"
):
    random.seed(0)
    model = select_model("openai_mini")
    instruction = "Convert the image to markdown. If there are any charts or diagrams, give a detailed description in its place."
    contents = []

    for p in tqdm(paths):
        doc = MultimodalDocument.load(p)
        o: MultimodalObject
        for o in random.sample(doc.objects, k=pages_per_doc):
            image = o.get_image()
            text = model.run([instruction, image])
            print(text)

            contents.append(load_doc_image(o.image_string, 512))
            contents.append(Paragraph(o.source))
            contents.append(Paragraph(text))
            contents.append(PageBreak())

    template = SimpleDocTemplate(str(path_out), pagesize=(8.5 * 72, 25.5 * 72))
    template.build(contents)
    print(Path(path_out).absolute())


"""
p analysis.py show_preds outputs/demo/acrv/openai_vision/clip_text/top_k_2.jsonl --path_out renders/demo_openai.pdf
p analysis.py show_preds outputs/demo/amlx/openai_vision/clip_text/top_k_10.jsonl --path_out renders/demo_openai_amlx.pdf
p analysis.py show_preds outputs/eval/openai/page/top_k=3.jsonl --path_out renders/openai_page_top_k=3.pdf
p analysis.py show_preds outputs/eval/openai/bm25_page/top_k=3.jsonl --path_out renders/openai_bm25_page_top_k=3.pdf
p analysis.py test_pdf_reader raw_data/annual_reports_2022_selected/NASDAQ_VERV_2022.pdf
p analysis.py test_load_from_pdf raw_data/annual_reports_2022_selected/NASDAQ_VERV_2022.pdf
p analysis.py test_load_from_excel_and_pdf raw_data/annual_reports_2022_selected/NASDAQ_VERV_2022.pdf
p analysis.py test_read_pdf_new

p analysis.py test_scores outputs/openai/colpali/top_k=5.json
p analysis.py test_scores outputs/claude/colpali/top_k=5.json
p analysis.py test_scores outputs/gemini/colpali/top_k=5.json
p analysis.py test_doc_content data/train/*.json
p analysis.py test_doc_parsing data/train/*.json
"""


if __name__ == "__main__":
    Fire()
