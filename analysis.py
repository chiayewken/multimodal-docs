from io import BytesIO

import requests
from PIL import Image
from fire import Fire
from sentence_transformers import SentenceTransformer, util


def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    raise ValueError(f"Failed to retrieve image. Status code: {response.status_code}")


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


if __name__ == "__main__":
    Fire()
