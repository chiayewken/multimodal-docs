import io
from pathlib import Path

import fitz  # imports the pymupdf library
from fire import Fire
from reportlab.platypus import (
    SimpleDocTemplate,
    Image as DocImage,
    Paragraph,
    PageBreak,
)
from sentence_transformers import SentenceTransformer, util

from data_loading import (
    load_image_from_url,
    convert_text_to_image,
    MultimodalSample,
    MultimodalData,
    MultimodalDocument,
)
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


def test_load_from_excel_and_pdf(
    path: str = "data/财报标注-0416.xlsx",
    pdf_dir: str = "raw_data/annual_reports_2022_selected",
):
    data = MultimodalData.load_from_excel_and_pdf(path, pdf_dir=pdf_dir)
    data.analyze()
    breakpoint()


"""
p analysis.py show_preds outputs/demo/acrv/openai_vision/clip_text/top_k_2.jsonl --path_out renders/demo_openai.pdf
p analysis.py show_preds outputs/demo/amlx/openai_vision/clip_text/top_k_10.jsonl --path_out renders/demo_openai_amlx.pdf
p analysis.py show_preds outputs/eval/openai/page/top_k=3.jsonl --path_out renders/openai_page_top_k=3.pdf
p analysis.py show_preds outputs/eval/openai/bm25_page/top_k=3.jsonl --path_out renders/openai_bm25_page_top_k=3.pdf
p analysis.py test_pdf_reader raw_data/annual_reports_2022_selected/NASDAQ_VERV_2022.pdf
p analysis.py test_load_from_pdf raw_data/annual_reports_2022_selected/NASDAQ_VERV_2022.pdf
p analysis.py test_load_from_excel_and_pdf raw_data/annual_reports_2022_selected/NASDAQ_VERV_2022.pdf
"""


if __name__ == "__main__":
    Fire()
