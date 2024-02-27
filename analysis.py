import io
from pathlib import Path

from fire import Fire
from sentence_transformers import SentenceTransformer, util

from data_loading import (
    load_image_from_url,
    convert_text_to_image,
    MultimodalSample,
    MultimodalData,
)

from reportlab.platypus import SimpleDocTemplate, Image as DocImage, Paragraph


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
    return DocImage(content, width=size, height=size)


def show_preds(path: str, path_out: str, image_size=256):
    Path(path_out).parent.mkdir(exist_ok=True, parents=True)
    doc = SimpleDocTemplate(str(path_out), pagesize=(5.5 * 72, 25.5 * 72))
    story = []

    sample: MultimodalSample
    for i, sample in enumerate(MultimodalData.load(path).samples):
        sample.print()
        story.append(Paragraph(f"<b>Prompt</b>:"))
        for x in sample.prompt.objects:
            if x.text:
                story.append(Paragraph(x.text))
            if x.image_string:
                story.append(load_doc_image(x.image_string, image_size))

        story.append(Paragraph(f"<b>Gold Answer</b>: {sample.answer}"))
        story.append(Paragraph(f"<b>Raw Output</b>: {sample.raw_output}"))

    doc.build(story)
    print(Path(path_out).absolute())


"""
p analysis.py show_preds outputs/demo/acrv/openai_vision/clip_text/top_k_2.jsonl --path_out renders/demo_openai.pdf
"""


if __name__ == "__main__":
    Fire()
