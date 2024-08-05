import random

from PIL import Image
from fire import Fire
from pydantic import BaseModel
from tqdm import tqdm

from data_loading import MultimodalDocument, MultimodalObject
from modeling import EvalModel, OpenAIMiniModel
from training import GemmaTrainer


class DocParser(BaseModel):
    def run(self, image: Image.Image) -> str:
        raise NotImplementedError


class MiniParser(DocParser):
    client: EvalModel = OpenAIMiniModel(max_output_tokens=2048)
    instruction: str = "Convert the image to markdown. If there are any charts or diagrams, give a detailed description in its place."

    def run(self, image: Image.Image) -> str:
        return self.client.run([self.instruction, image])


def generate_data(*paths, path_out: str, pages_per_doc: int = 50):
    random.seed(0)
    parser = MiniParser()
    contents = []
    num_tokens = 0

    for p in tqdm(paths):
        doc = MultimodalDocument.load(p)
        o: MultimodalObject
        for o in random.sample(doc.objects, k=pages_per_doc):
            image = o.get_image()
            o.text = parser.run(image)
            contents.append(o)
            num_tokens += len(o.text) // 4
            print(o.text, dict(average_tokens=num_tokens // len(contents)))

        MultimodalDocument(objects=contents).save(path_out)


def train_gemma(path: str, **kwargs):
    data = MultimodalDocument.load(path)
    trainer = GemmaTrainer(**kwargs)
    samples = [([o.get_image()], o.text) for o in data.objects]
    trainer.run(samples)


"""
p parsing.py generate_data data/train/*.json --path_out outputs/parse/train.json
p parsing.py generate_data data/test/*.json --path_out outputs/parse/test.json --pages_per_doc 5
p parsing.py train_gemma outputs/parse/test.json --save_dir outputs/train_parse/gemma --epochs 3
"""


if __name__ == "__main__":
    Fire()
