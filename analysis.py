import hashlib
import io
import json
import random
import re
from ast import literal_eval
from collections import Counter
from itertools import takewhile
from pathlib import Path
from typing import List

# noinspection PyPackageRequirements
import fitz  # imports the pymupdf library
import krippendorff
import numpy as np
import pandas as pd
import tiktoken
from datasets import load_dataset
from fire import Fire
from reportlab.platypus import (
    SimpleDocTemplate,
    Image as DocImage,
    Paragraph,
    PageBreak,
)
from scipy import stats
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
    get_domain,
    load_valid_questions,
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


def read_companies(path: str = "data/nyse.txt", seed: int = 0):
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
            records.append(
                dict(
                    name=a.strip(),
                    search_url=f"https://www.annualreports.com/Companies?search={'+'.join(a.split(',')[0].split())}",
                    industry=b.strip(),
                    sector=c.strip(),
                )
            )

    df = pd.DataFrame(records)
    print(df.shape)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
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


def test_content_distribution(*paths: str):
    records = []
    tokenizer = tiktoken.get_encoding("cl100k_base")

    for p in tqdm(paths):
        doc = MultimodalDocument.load(p)
        categories = [o.category for page in doc.pages for o in page.objects]
        text = "".join(p.text for p in doc.pages).replace("<|endoftext|>", "")
        info = dict(
            source=p,
            domain=doc.get_domain(),
            figure=categories.count("Picture"),
            table=categories.count("Table"),
            tokens=len(tokenizer.encode(text)),
        )
        records.append(info)

    df = pd.DataFrame(records)
    print(df["domain"].value_counts())
    print(df.groupby("domain").mean(numeric_only=True).round(1))
    print(df.mean(numeric_only=True).round(1))


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
    records = []
    for p in tqdm(paths):
        doc = fitz.open(p)
        records.append(dict(path=p, pages=doc.page_count, size=Path(p).stat().st_size))

    df = pd.DataFrame(records)
    df = df.sort_values("pages", ascending=False)
    average = df["pages"].mean()
    print(df)
    print(df.shape, dict(average=average))


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


def plot_data_chart(path: str = "data/test/metadata.csv", path_out: str = "chart.png"):
    import plotly.graph_objects as go

    df = pd.read_csv(path)
    print(df.shape)
    categories = {}
    for url, label in df.values:
        domain = get_domain(url)
        categories.setdefault(domain, []).append(label)

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


def plot_multimodal_chart(*paths: str, path_out: str = "chart.png"):
    import plotly.graph_objects as go

    categories = {}
    for p in tqdm(paths):
        doc = MultimodalDocument.load(p)
        domain = doc.get_domain()
        for page in doc.pages:
            if len(page.text) > 20:
                categories.setdefault(domain, []).append("Text")
            object_labels = [o.category for o in page.get_tables_and_figures()]
            if "Table" in object_labels:
                categories.setdefault(domain, []).append("Table")
            if "Picture" in object_labels:
                categories.setdefault(domain, []).append("Figure")

    categories = {key: random.sample(lst, 1000) for key, lst in categories.items()}
    print({key: Counter(lst) for key, lst in categories.items()})

    values = [len(lst) for lst in categories.values()]
    labels = list(categories.keys())
    parents = [""] * len(categories)
    for i, (key, lst) in enumerate(categories.items()):
        unique = sorted(set(lst))
        values.extend([lst.count(u) for u in unique])
        categories[key] = unique
        labels.extend([" " * i + u for u in unique])
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


def test_languages(*paths: str, model_name: str = "langdetect"):
    model = select_model(model_name)
    for p in paths:
        doc = MultimodalDocument.load(p)
        texts = [p.text for p in doc.pages]
        languages = [model.run([t]) for t in texts]
        old_objects = [o for p in doc.pages for o in p.get_tables_and_figures()]
        new_objects = [
            o
            for i, p in enumerate(doc.pages)
            if languages[i] == "en"
            for o in p.get_tables_and_figures()
        ]
        print(
            dict(
                p=p,
                pages=len(doc.pages),
                objects_keep=round(len(new_objects) / len(old_objects), 2),
                languages=Counter(languages),
            )
        )


def test_judge_agreement(*paths: str):
    for p in paths:
        data = MultimodalData.load(p)
        num_judges = len(data.samples[0].judgements)
        scores = [[] for _ in range(num_judges)]
        if not data.samples[0].judgements:
            print(dict(skip=p))
            continue

        for sample in data.samples:
            assert len(sample.judgements) == num_judges
            for i, judge in enumerate(sample.judgements):
                scores[i].append(judge.score)

        alpha = krippendorff.alpha(
            reliability_data=np.array(scores), level_of_measurement="ordinal"
        )
        print(dict(path=p, krippendorff_alpha=round(alpha, 3)))


def test_judge_self_bias(*paths: str):
    for p in paths:
        data = MultimodalData.load(p)
        judges = set(j.name for s in data.samples for j in s.judgements)
        if data.samples[0].generator not in judges:
            print(dict(skip=p))
            continue

        scores_self = []
        scores_other = []
        for sample in data.samples:
            for judge in sample.judgements:
                if judge.name == sample.annotator:
                    scores_self.append(judge.score)
                else:
                    scores_other.append(judge.score)

        print(
            dict(
                path=p,
                self=round(sum(scores_self) / len(scores_self), 3),
                other=round(sum(scores_other) / len(scores_other), 3),
            )
        )


def content_filter_fn(
    samples: List[MultimodalSample], category: str
) -> List[MultimodalSample]:
    if category == "text":
        return [s for s in samples if "text" in s.category]
    elif category == "figure":
        return [s for s in samples if "figure" in s.category]
    elif category == "table":
        return [s for s in samples if "table" in s.category]
    elif category == "academic":
        return [s for s in samples if category in get_domain(s.source).lower()]
    elif category == "finance":
        return [s for s in samples if "report" in get_domain(s.source).lower()]
    elif category == "product":
        return [s for s in samples if "technical" in get_domain(s.source).lower()]
    else:
        return samples


def remove_common_affix(texts):
    if not texts:
        return []

    # Find common prefix
    def common_prefix(s1, s2):
        return "".join(
            c[0] for c in takewhile(lambda x: all(x[0] == y for y in x), zip(*[s1, s2]))
        )

    # Find common suffix
    def common_suffix(s1, s2):
        return common_prefix(s1[::-1], s2[::-1])[::-1]

    # Get common prefix and suffix for all strings
    prefix = texts[0]
    suffix = texts[0]
    for s in texts[1:]:
        prefix = common_prefix(prefix, s)
        suffix = common_suffix(suffix, s)

    # Remove prefix and suffix from each string
    result = [s[len(prefix) : -len(suffix) or None] for s in texts]
    return result


def test_results(*paths: str, sort_key="all", limit: int = 0, valid_path: str = ""):
    records = []
    valid_set = set()
    if valid_path:
        valid_set = load_valid_questions(valid_path)
        print(dict(valid_question_set=len(valid_set)))

    for p in paths:
        info = dict(path=p)
        if not MultimodalData.load(p).samples[0].judgements:
            continue
        for label in [
            "academic",
            "product",
            "finance",
            "text",
            "figure",
            "table",
            "all",
        ]:
            data = MultimodalData.load(p)
            if valid_path:
                data.samples = [s for s in data.samples if s.question in valid_set]
            if limit > 0:
                data.samples = data.samples[:limit]
            scores = [
                judge.score
                for s in content_filter_fn(data.samples, label)
                for judge in s.judgements
            ]
            info[label] = sum(scores) / len(scores)
        records.append(info)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)
    df = pd.DataFrame(records).sort_values(sort_key).reset_index(drop=True)
    df["path"] = remove_common_affix(df["path"].tolist())
    print(df.round(2))


def length_fn(length: int) -> str:
    if length <= 1:
        return "0-1"
    elif length <= 3:
        return "2-3"
    elif length <= 5:
        return "4-5"
    elif length <= 7:
        return "6-7"
    elif length <= 9:
        return "8-9"
    else:
        return "10+"


def test_results_by_multimodal_length(*paths: str):
    records = []

    for p in paths:
        if not MultimodalData.load(p).samples[0].judgements:
            continue

        data = MultimodalData.load(p)
        documents = data.load_documents()

        for s in tqdm(data.samples):
            score = sum(j.score for j in s.judgements) / len(s.judgements)
            doc = documents[s.source]
            pages = [page for page in doc.pages if page.number in s.retrieved_pages]
            assert len(pages) == 5
            length = sum(len(page.get_tables_and_figures()) for page in pages)
            records.append(dict(path=p, length=length, score=score))

    df = pd.DataFrame(records)
    print(df.shape)
    print(df.round(2).sample(10))

    df["length"] = df["length"].apply(length_fn)
    groups = df.groupby("length").agg({"score": ["mean", "count"]}).reset_index()
    groups.columns = ["length", "average_score", "group_size"]
    groups = groups.sort_values("length")
    print(groups.round(2))


def test_retriever_results(*paths: str, metric="mrr"):
    records = []

    for p in paths:
        info = dict(path=p)
        for label in ["text", "figure", "table", "all"]:
            data = MultimodalData.load(p)
            scores = []

            for sample in content_filter_fn(data.samples, label):
                sorted_ids = sample.retrieved_pages
                assert len(sample.evidence_pages) == 1
                rank = sorted_ids.index(sample.evidence_pages[0])
                if metric == "mrr":
                    scores.append(1 / (rank + 1))
                elif metric == "top-5-recall":
                    scores.append(int(rank < 5))
                else:
                    raise ValueError(f"Invalid metric: {metric}")

            info[label] = sum(scores) / len(scores) * 100
        records.append(info)

    df = pd.DataFrame(records)
    df = df.sort_values("all")
    print(df.round(1))
    print(dict(metric=metric))


def show_model_preds(path: str, path_out: str, num_sample: int = 30):
    data = MultimodalData.load(path)
    content = []
    mapping = {}
    random.seed(0)
    records = []

    sample: MultimodalSample
    if num_sample > 0:
        data.samples = random.sample(data.samples, k=num_sample)
    for sample in tqdm(data.samples):
        if sample.source not in mapping:
            mapping[sample.source] = MultimodalDocument.load(sample.source)
        doc = mapping[sample.source]
        pages = [p for p in doc.pages if p.number in sample.evidence_pages]
        assert len(pages) == 1
        question_id = hash_text(sample.source + sample.question)
        content.extend(
            [
                f"<b>Source</b> ({sample.category}): {sample.source} (Page {sample.evidence_pages[0]})",
                f"<b>Question</b> ({sample.annotator}): {sample.question}",
                f"<b>Question ID</b> {question_id}",
                f"<b>Model Response</b> ({sample.generator}): {sample.pred.replace('<', '>')}",
                *[f"<b>Judge</b> ({j.name}): {j.score}" for j in sample.judgements],
                pages[0].get_full_image(),
                "",
            ]
        )

        records.append(
            dict(
                domain=get_domain(sample.source),
                content_category=sample.category,
                question_id=question_id,
                question=sample.question,
                pred=sample.pred,
                judge_scores=[j.score for j in sample.judgements],
                human_score="",
            )
        )

    save_multimodal_document(content, path_out, pagesize=(595, 595 * 2))
    df = pd.DataFrame(records)
    print(df.shape)
    print(df.head())
    df.to_excel(Path(path_out).with_suffix(".xlsx"), index=False)


def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def prepare_question_sheet(
    path: str,
    path_out: str,
    num_sample: int = 30,
    domains: List[str] = (),
    data_file: str = "",
):
    data = MultimodalData.load(path)
    documents = {}
    records = []
    content = []

    if num_sample > 0:
        random.seed(0)
        data.samples = random.sample(data.samples, k=num_sample)
    sample: MultimodalSample
    for sample in tqdm(data.samples):
        info = dict(
            data_file=data_file,
            domain=get_domain(sample.source),
            content_category=sample.category,
            question_id=hash_text(sample.source + sample.question),
            question=sample.question,
            check_1="",
            check_2="",
            check_3="",
            check_4="",
        )
        if domains and info["domain"] not in domains:
            continue
        records.append(info)

        if sample.source not in documents:
            documents[sample.source] = MultimodalDocument.load(sample.source)
        doc = documents[sample.source]
        pages = [p for p in doc.pages if p.number in sample.evidence_pages]
        assert len(pages) == 1
        content.extend(
            [
                f"<b>Source</b> ({sample.category}): {sample.source} (Page {sample.evidence_pages[0]})",
                f"<b>Question</b> ({sample.annotator}): {sample.question}",
                f"<b>Question ID</b> {info['question_id']}",
                pages[0].get_full_image(),
                "",
            ]
        )

    Path(path_out).parent.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(records)
    print(df.shape)
    print(df.head())
    df.to_excel(path_out, index=False)
    save_multimodal_document(content, data_file, pagesize=(595, 595 * 2))


def check_excel(*paths: str):
    for p in paths:
        df = pd.read_excel(p)
        print(p, df.shape)
        print(df["content_category"].value_counts())
        print(df.head())


def test_product_domain(path: str):
    data = MultimodalData.load(path)
    samples = []
    seen = set()
    for s in data.samples:
        key = s.source + s.question
        print(s.source)
        if "manuals" in get_domain(s.source).lower() and key not in seen:
            seen.add(key)
            samples.append(s)
            s.retrieved_pages = []
    print(dict(samples=len(samples)))
    MultimodalData(samples=samples).save(path)


def test_read_pdfs(*paths: str):
    for p in tqdm(paths):
        print(p)
        doc = fitz.open(p)
        lst = [len(page.get_text()) for page in doc.pages()]
        print(dict(min=min(lst), max=max(lst), average=sum(lst) / len(lst)))


def check_duplicate_questions(*paths: str):
    questions = []
    evidence = []
    for p in paths:
        data = MultimodalData.load(p)
        for s in data.samples:
            questions.append(s.question)
            evidence.append((s.source, s.evidence_pages[0]))

    print(dict(total=len(questions), unique=len(set(questions))))
    print(dict(total_evidence=len(evidence), unique_evidence=len(set(evidence))))


def test_question_distribution(*paths, valid_path: str):
    valid_set = []
    if valid_path:
        valid_set = load_valid_questions(valid_path)

    records = []
    for p in paths:
        data = MultimodalData.load(p)
        if valid_path:
            data.samples = [s for s in data.samples if s.question in valid_set]
        for sample in data.samples:
            info = dict(
                domain=get_domain(sample.source),
                category=sample.category,
                question=sample.question,
            )
            records.append(info)

    df = pd.DataFrame(records)
    for group in df.groupby("domain"):
        print(dict(domain=group[0], total=group[1].shape[0]))
        for subgroup in group[1].groupby("category"):
            print(dict(category=subgroup[0], total=subgroup[1].shape[0]))


def test_human_agreement(path: str):
    df = pd.read_excel(path)
    df = df.dropna(subset=["human_score"])
    judge_scores = df["judge_scores"].apply(literal_eval).apply(np.mean).tolist()
    human_scores = df["human_score"].tolist()
    correlation, p_value = stats.pearsonr(
        np.array(judge_scores), np.array(human_scores)
    )
    print(dict(correlation=correlation, p_value=p_value))

    # Plot the distribution of scores between human and model scores
    counts_judge = [0] * 5
    counts_human = [0] * 5
    for score in judge_scores:
        counts_judge[round(score) - 1] += 1
    for score in human_scores:
        counts_human[round(score) - 1] += 1

    print(dict(judge=counts_judge, human=counts_human))

    dist_judge = [round(x / sum(counts_judge) * 100, 1) for x in counts_judge]
    dist_human = [round(x / sum(counts_human) * 100, 1) for x in counts_human]
    print(dict(judge=dist_judge, human=dist_human))

    print(dict(avg_judge=np.mean(judge_scores), avg_human=np.mean(human_scores)))
    breakpoint()


def test_question_types(
    path: str = "data/annotation/score_checking_100_hp.xlsx",
    path_out: str = "data/annotation/question_types_100.csv",
):
    df = pd.read_excel(path)
    print(df.shape)

    model = select_model("claude")
    instruction = "Based on the question, assign category labels from the following list. If a question is applicable to multiple labels, separate the labels with commas. Give only the labels without any additional information."
    instruction += "\n1. Analytical Reasoning and Pattern Recognition: Questions about trends, comparisons, and implications (e.g., engagement trends, performance trends)"
    instruction += "\n2. Technical Analysis: Questions about specific technical details (e.g., UEFI BIOS, shutter speeds, X-sync speeds) and applications of technical concepts."
    instruction += "\n3. Commonsense and Domain Knowledge: Questions requiring general knowledge or background knowledge in fields such as finance, cybersecurity, photography."
    instruction += "\n4. Predictive Analysis: Questions about expected outcomes or forecasting changes based on given information."
    instruction += "\n5. Visual Interpretation: Questions based on interpreting icons, diagrams, or charts."
    instruction += "\n6. Mathematical Reasoning: Questions involving mathematical concepts or calculation from data and tables."
    instruction += "\nExample question: In the provided diagram, the set \( B \) is represented by a vertical line intersecting the triangle at points \( p \) and \( q \). Given that \( p \) and \( q \) are maximal elements in \( B \) but \( \argmax_B H = \{p\} \), explain why \( q \) is not obtained via the maximum entropy principle and discuss the implications of this for the optimization of injective monotones in preordered spaces."
    instruction += "\nExample labels: Mathematical Reasoning"
    instruction += "\nQuestion: "

    df["label"] = ""
    counts = {}

    for i, row in tqdm(df.iterrows()):
        question = row["question"]
        if not row["label"]:
            if "figure" in row["content_category"]:
                question = f"Based on the figure or diagram: {question}"
            if "table" in row["content_category"]:
                question = f"Based on the table image: {question}"
            result = model.run([instruction + question])
            result = result.split(":")[-1].strip()
            labels = [x.strip() for x in result.split(",") if x.strip()]
            print(dict(question=question, result=result, labels=labels))
            print(counts)
            df.at[i, "label"] = ", ".join(labels)
            for lab in labels:
                counts[lab] = counts.get(lab, 0) + 1

    print(dict(counts=counts, total=sum(counts.values())))
    df.to_csv(path_out, index=False)


def test_question_self_bias(*paths: str):
    for p in paths:
        data = MultimodalData.load(p)

        scores_self = []
        scores_other = []
        for sample in data.samples:
            score = np.mean([j.score for j in sample.judgements])
            model_name = Path(p).parts[1].split("-")[0]
            if model_name in sample.annotator:
                scores_self.append(score)
            else:
                scores_other.append(score)

        print(dict(path=p, self=np.mean(scores_self), other=np.mean(scores_other)))


def export_question_types(
    path: str = "data/annotation/question_types_100.csv",
    path_out: str = "data/annotation/question_types_100.xlsx",
):
    df = pd.read_csv(path)
    df = df[["domain", "content_category", "question", "label"]]
    df.to_excel(path_out, index=False)


def test_pairwise_judge_agreement(path: str):
    df = pd.read_excel(path)
    df = df.dropna(subset=["human_score"])
    judge_scores = df["judge_scores"].apply(literal_eval).tolist()
    list1 = [row[0] for row in judge_scores]
    list2 = [row[1] for row in judge_scores]
    list3 = [row[2] for row in judge_scores]

    values = []
    for scores_1 in [list1, list2, list3]:
        for scores_2 in [list1, list2, list3]:
            if scores_1 == scores_2:
                continue
            correlation, p_value = stats.pearsonr(scores_1, scores_2)
            values.append(correlation)
            print(dict(correlation=correlation, p_value=p_value))
    print(dict(pairs=len(values)))
    print(dict(average_correlation=np.mean(values)))


def compare_qwen_answers(path_a: str, path_b: str, path_out: str):
    data_a = MultimodalData.load(path_a)
    data_b = MultimodalData.load(path_b)
    assert len(data_a.samples) == len(data_b.samples)

    total = 0
    content = []
    for a, b in tqdm(list(zip(data_a.samples, data_b.samples))):
        score_a = round(np.mean([j.score for j in a.judgements]), 2)
        score_b = round(np.mean([j.score for j in b.judgements]), 2)
        assert a.question == b.question
        page_numbers = sorted(b.retrieved_pages[:5])

        if score_b - score_a > 1.0 and b.evidence_pages[0] in page_numbers:
            doc = MultimodalDocument.load(b.source)
            pages = [p for p in doc.pages if p.number in page_numbers]
            content.extend(
                [
                    f"<b>Source</b> ({b.category}): {b.source} (Page {b.evidence_pages[0]})",
                    f"<b>Retrieved Pages</b>: {b.retrieved_pages}",
                    f"<b>Question</b> ({b.annotator}): {b.question}",
                    f"<b>Answer A</b> ({score_a}): {a.pred}",
                    f"<b>Answer B</b> ({score_b}): {b.pred}",
                    *[page.get_full_image() for page in pages],
                    "",
                ]
            )
            total += 1
            if total > 20:
                break

    save_multimodal_document(content, path_out, pagesize=(595, 595 * 2))


def test_question_lengths(*paths: str):
    question_lengths = []
    answer_lengths = []
    for p in paths:
        data = MultimodalData.load(p)
        for sample in data.samples:
            question_lengths.append(len(sample.question.split()))
            answer_lengths.append(len(sample.pred.split()))

    print(np.mean(question_lengths), np.mean(answer_lengths))


def test_mmlongbench_lengths(path: str = "data/mmlongbenchdoc.json"):
    question_lengths = []
    answer_lengths = []
    with open(path) as f:
        data = json.load(f)

    print(len(data))
    for item in data:
        question = item["question"]
        answer = item["answer"]
        question_lengths.append(len(question.split()))
        answer_lengths.append(len(answer.split()))

    print(np.mean(question_lengths), np.mean(answer_lengths))


def test_slidevqa_lengths(path: str = "data/slidevqa.jsonl"):
    question_lengths = []
    answer_lengths = []

    with open(path) as f:
        for line in f:
            item = json.loads(line)
            question = item["question"]
            answer = item["answer"]
            question_lengths.append(len(question.split()))
            answer_lengths.append(len(answer.split()))

    print(np.mean(question_lengths), np.mean(answer_lengths))


def test_docvqa_lengths():
    question_lengths = []
    answer_lengths = []

    data = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    for sample in tqdm(data):
        question_lengths.append(len(sample["question"].split()))
        for answer in sample["answers"]:
            answer_lengths.append(len(answer.split()))

    print(np.mean(question_lengths), np.mean(answer_lengths))


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

p analysis.py test_languages data/test/*.json

p analysis.py test_judge_agreement outputs/*/colpali/top_k=5.json
{'path': 'outputs/claude-3-5-sonnet-20240620/colpali/top_k=5.json', 'krippendorff_alpha': 0.634}
{'path': 'outputs/gemini-1.5-pro-001/colpali/top_k=5.json', 'krippendorff_alpha': 0.523}
{'path': 'outputs/gpt-4o-2024-08-06/colpali/top_k=5.json', 'krippendorff_alpha': 0.594}
{'path': 'outputs/idefics/colpali/top_k=5.json', 'krippendorff_alpha': 0.558}
{'path': 'outputs/intern/colpali/top_k=5.json', 'krippendorff_alpha': 0.689}
{'path': 'outputs/onevision/colpali/top_k=5.json', 'krippendorff_alpha': 0.674}

p analysis.py test_judge_self_bias outputs/*/colpali/top_k=5.json
{'path': 'outputs/claude-3-5-sonnet-20240620/colpali/top_k=5.json', 'self': 4.556, 'other': 4.517}
{'path': 'outputs/gemini-1.5-pro-001/colpali/top_k=5.json', 'self': 4.311, 'other': 4.417}
{'path': 'outputs/gpt-4o-2024-08-06/colpali/top_k=5.json', 'self': 4.556, 'other': 4.528}

p analysis.py plot_multimodal_chart data/test/*.json
p analysis.py test_results outputs/*/colpali/top_k=5.json
bash scripts/eval_retrievers.sh
p analysis.py test_retriever_results outputs/retrieve/test/*.json
p analysis.py show_model_preds outputs/claude-3-5-sonnet-20240620/colpali/top_k\=5.json renders/pred_claude.pdf
p analysis.py show_model_preds outputs/claude-3-5-sonnet-20240620/colpali/top_k\=5.json data/annotation/demo.pdf

p analysis.py prepare_question_sheet outputs/claude-3-5-sonnet-20240620/colpali/top_k\=5.json data/annotation/demo.xlsx --domains "Financial<br>Report,Technical<br>Manuals" --data_file data/annotation/demo.pdf
p analysis.py prepare_question_sheet data/questions/test_finance.json data/annotation/finance.xlsx --data_file data/annotation/finance.pdf --num_sample 0
p analysis.py prepare_question_sheet data/questions/test_academic.json data/annotation/academic.xlsx --data_file data/annotation/academic.pdf --num_sample 0
p analysis.py prepare_question_sheet data/questions/test_product.json data/annotation/product.xlsx --data_file data/annotation/product.pdf --num_sample 0

p analysis.py test_document_lengths data/test/24*.pdf
p analysis.py test_document_lengths data/test/NY*.pdf
p analysis.py check_excel data/annotation/*.xlsx
p analysis.py test_content_distribution data/test/*.json
p analysis.py test_content_distribution data/test/NY*.json
p analysis.py plot_data_chart
p analysis.py test_product_domain data/questions/test_product.json
p analysis.py test_read_pdfs data/train/*.pdf
p analysis.py check_duplicate_questions data/questions/train.json
p analysis.py check_duplicate_questions data/questions/train.json data/questions/train2.json
p analysis.py check_duplicate_questions data/questions/train.json data/questions/train3.json

p analysis.py show_model_preds outputs/qwen/colpali_sample_100/top_k=5.json data/annotation/score_checking_100.pdf --num_sample 0
p analysis.py test_judge_agreement outputs/qwen/colpali_sample_100/top_k=5.json
p analysis.py test_judge_self_bias outputs/*/colpali/top_k=5.json
p analysis.py test_judge_agreement outputs/*/colpali/top_k=5.json
p analysis.py test_question_distribution outputs/azure/colpali/top_k=5.json --valid_path data/annotation/valid_questions.json
p analysis.py test_human_agreement data/annotation/score_checking_100_hp.xlsx
p analysis.py test_pairwise_judge_agreement data/annotation/score_checking_100_hp.xlsx
p analysis.py compare_qwen_answers outputs/qwen/colpali/top_k=5.json outputs/swift_qwen_10k/colpali/top_k=5.json --path_out renders/qwen_vs_ours.pdf
p analysis.py test_question_lengths outputs/swift_qwen/colpali/top_k=5.json
p analysis.py test_question_self_bias outputs/*/colpali/top_k=5.json
"""


if __name__ == "__main__":
    Fire()
