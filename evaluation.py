from pathlib import Path

from fire import Fire
from tqdm import tqdm

from data_loading import MultimodalData, MultimodalObject, MultimodalDocument
from modeling import select_model
from retrieval import select_retriever


def generate_answers(
    data_path: str,
    generator_name: str,
    retriever_name: str,
    output_dir: str = "outputs",
    top_k: int = 5,
):
    generator = select_model(generator_name)
    retriever = select_retriever(retriever_name, top_k=top_k)
    data = MultimodalData.load(data_path)
    path_out = Path(output_dir, generator_name, retriever_name, f"{top_k=}.json")
    Path(path_out).parent.mkdir(exist_ok=True, parents=True)

    with open(path_out, "w") as f:
        for sample in tqdm(data.samples, desc=str(path_out)):
            doc = MultimodalDocument.load(sample.source)
            query = MultimodalObject(text=sample.question)
            pages = retriever.run(query, doc).objects
            sample.retrieved_pages = [p.page for p in pages]

            prompt = f"Answer the following question in 200 words or less: {query.text}"
            inputs = [prompt] + [p.get_image() or p.text for p in pages]
            sample.pred = generator.run(inputs)
            sample.generator = generator_name
            print(sample.json(indent=2))
            print(sample.json(), file=f)


"""
p evaluation.py generate_answers data/questions.json --generator_name openai --retriever_name bm25
"""


if __name__ == "__main__":
    Fire()
