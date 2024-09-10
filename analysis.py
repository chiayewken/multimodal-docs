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
    save_multimodal_document,
)
from modeling import select_model
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

    if image.width > image.height:
        new_width = size
        new_height = int(size * image.height / image.width)
    else:
        new_height = size
        new_width = int(size * image.width / image.height)

    return DocImage(content, width=new_width, height=new_height)


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


def read_brands(path: str = "data/brands.txt", seed: int = 0):
    # https://www.manualslib.com/brand/
    records = []
    random.seed(0)

    with open(path) as f:
        for line in f:
            items = [x.strip() for x in line.strip().split(",")]
            records.append(dict(name=items[0], categories=items[1:]))
            random.shuffle(records[-1]["categories"])

    df = pd.DataFrame(records)
    print(df.shape)
    pd.set_option("display.max_rows", None)
    print(df.sample(n=20, random_state=seed))


def test_read_pdf_new(path: str = "data/reports/NYSE_HI_2023.pdf"):
    doc = MultimodalDocument.load_from_pdf(path)
    path_out = Path("renders", Path(path).name)
    Path(path_out).parent.mkdir(exist_ok=True, parents=True)

    story = []
    random.seed(0)
    objects = sorted(random.sample(doc.objects, 10), key=lambda o: o.page)
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


def show_document(path: str, output_dir="renders"):
    doc = MultimodalDocument.load(path)
    path_out = Path(output_dir, Path(path).name).with_suffix(".pdf")
    Path(path_out).parent.mkdir(exist_ok=True, parents=True)

    story = []
    sample: MultimodalSample
    for page in tqdm(doc.pages):
        story.append(Paragraph(f"<b>Page {page.number}</b>:"))
        story.append(Paragraph(f"<b>Source</b>: {page.source}"))
        story.append(Paragraph(f"<b>Full Image:</b>"))
        story.append(load_doc_image(page.image_string, 256))
        story.append(Paragraph(f"<b>Full Text:</b>"))
        try:
            story.append(Paragraph(page.text))
        except Exception as e:
            print(e, page.text)

        story.append(Paragraph(f"<b>Objects:</b>"))
        for i, o in enumerate(page.get_tables_and_figures()):
            story.append(Paragraph(f"<b>Object {i} ({o.category})</b>:"))
            story.append(load_doc_image(o.image_string, 256))
        story.append(PageBreak())

    template = SimpleDocTemplate(str(path_out), pagesize=(8.5 * 72, 25.5 * 72))
    template.build(story)
    print(Path(path_out).absolute())


def test_object_categories(*paths: str):
    docs = [MultimodalDocument.load(p) for p in tqdm(paths)]
    for label in ["Text", "Picture", "Table"]:
        print(f"How many pages have {label}?")
        count = sum(
            any(o.category == label for o in p.objects) for d in docs for p in d.pages
        )
        total = sum(len(d.pages) for d in docs)
        print(count / total)


def test_document_lengths(*paths: str):
    print("Finance")
    docs = [MultimodalDocument.load(p) for p in tqdm(paths) if "NYSE" in p]
    print(sum(len(d.pages) for d in docs) / len(docs))
    print("Academic")
    docs = [MultimodalDocument.load(p) for p in tqdm(paths) if "NYSE" not in p]
    print(sum(len(d.pages) for d in docs) / len(docs))


def test_questions(path: str, path_out="demo.pdf", num_sample: int = 30):
    data = MultimodalData.load(path)
    content = []
    mapping = {}
    random.seed(0)

    sample: MultimodalSample
    for sample in tqdm(random.sample(data.samples, k=num_sample)):
        if sample.source not in mapping:
            mapping[sample.source] = MultimodalDocument.load(sample.source)
        doc = mapping[sample.source]
        pages = [p for p in doc.pages if p.number in sample.evidence_pages]
        assert len(pages) == 1
        content.extend(
            [
                f"Category: {sample.category}",
                f"Annotator: {sample.annotator}",
                f"Source: {sample.source} (Page {sample.evidence_pages[0]})",
                f"Question: {sample.question}",
                pages[0].get_full_image(),
                "",
            ]
        )

    save_multimodal_document(content, path_out)


def test_retrieval(data_path: str):
    data = MultimodalData.load(data_path)
    for k in range(1, 100):
        success = []
        for sample in data.samples:
            assert len(sample.evidence_pages) == 1
            success.append(sample.evidence_pages[0] in sample.retrieved_pages[:k])
        print(dict(k=k, success=sum(success) / len(success)))


def plot_data_chart(path_out: str = "chart.png"):
    import plotly.graph_objects as go

    # New data
    categories = {
        "Financial<br>Report": [
            "healthcare",
            "materials",
            "consumer",
            "financial",
            "industrial",
            "services",
            "consumer",
            "materials",
            "technology",
            "technology",
        ],
        "Academic<br>Paper": [
            "mathematics",
            "biology",
            "physics",
            "computer",
            "computer",
            "physics",
            "finance",
            "physics",
            "statistics",
            "engineering",
        ],
        "Technical<br>Manuals": [
            "phone",
            "stove",
            "router",
            "blender",
            "vacuum",
            "breaker",
            "laptop",
            "fridge",
            "car",
            "phone",
        ],
    }

    values = [len(lst) for lst in categories.values()]
    labels = list(categories.keys())
    parents = [""] * len(categories)
    for key, lst in categories.items():
        unique = sorted(set(lst))
        values.extend([lst.count(u) for u in unique])
        categories[key] = unique
        labels.extend(unique)
        parents.extend([key] * len(unique))

    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            insidetextorientation="radial",  # This makes labels follow the slice angle
            textfont=dict(size=10, color="white"),  # Set font color to white
        )
    )

    # Update layout for better visualization and fixed aspect ratio
    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        width=400,  # Set the width of the figure
        height=400,  # Set the height of the figure
    )

    fig.show()
    fig.write_image(path_out, scale=4)


"""
p analysis.py test_pdf_reader raw_data/annual_reports_2022_selected/NASDAQ_VERV_2022.pdf
p analysis.py test_load_from_pdf raw_data/annual_reports_2022_selected/NASDAQ_VERV_2022.pdf
p analysis.py test_load_from_excel_and_pdf raw_data/annual_reports_2022_selected/NASDAQ_VERV_2022.pdf
p analysis.py test_read_pdf_new

p analysis.py test_scores outputs/openai/colpali/top_k=5.json
p analysis.py test_scores outputs/claude/colpali/top_k=5.json
p analysis.py test_scores outputs/gemini/colpali/top_k=5.json
p analysis.py test_doc_content data/train/*.json
p analysis.py test_doc_parsing data/train/*.json
p analysis.py show_document data/train/2012.14143v1.json
p analysis.py test_object_categories data/train/*.json
p analysis.py test_document_lengths data/test/*.json
p analysis.py test_questions data/questions/test.json

p analysis.py test_retrieval data/questions/test.json
{'k': 1, 'success': 0.6896551724137931}                                                                                                                   
{'k': 2, 'success': 0.7586206896551724}    
{'k': 3, 'success': 0.8103448275862069}                                                                                                                   
{'k': 4, 'success': 0.8448275862068966} 
{'k': 5, 'success': 0.8620689655172413}                                                                                                                   
{'k': 6, 'success': 0.8620689655172413}                                                                                                                   
{'k': 7, 'success': 0.8793103448275862}                                                                                                                   
{'k': 8, 'success': 0.8793103448275862}                                                                                                                   
{'k': 9, 'success': 0.8793103448275862}                                                                                                                   
{'k': 10, 'success': 0.896551724137931}  
"""


if __name__ == "__main__":
    Fire()
