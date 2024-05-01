import json
from pathlib import Path

from fire import Fire
from tqdm import tqdm

from data_loading import MultimodalData, MultimodalObject
from modeling import select_model
from retrieval import select_retriever


def main(
    path: str = "data/财报标注-0416.xlsx",
    pdf_dir: str = "raw_data/annual_reports_2022_selected",
    generator_name: str = "openai",
    retriever_name: str = "page",
    top_k: int = 3,
    output_dir: str = "outputs/eval",
    **kwargs,
):
    print(locals())
    data = MultimodalData.load_from_excel_and_pdf(path, pdf_dir)
    generator = select_model(generator_name)
    retriever = select_retriever(retriever_name, top_k=top_k, **kwargs)
    path_out = Path(output_dir, generator_name, retriever_name, f"{top_k=}.jsonl")

    progress = tqdm(data.samples, desc=str(path_out))
    retrieve_success = []
    for sample in progress:
        query = MultimodalObject(text=sample.question)
        sample.prompt = retriever.run(query, doc=sample.doc)
        sample.prompt.objects.insert(0, query)

        # Avoid "no image in input" error
        if all(x.image_string == "" for x in sample.prompt.objects):
            sample.prompt.objects.append(
                [x for x in sample.doc.objects if x.image_string][0]
            )

        sample.raw_output = generator.run(sample.prompt)
        info = dict(
            source=sample.doc.source,
            question=sample.question,
            answer=sample.answer,
            raw_output=sample.raw_output,
            evidence=sample.evidence.objects[0].source,
            retrieve_success=sample.evidence.objects[0].page
            in [o.page for o in sample.prompt.objects],
        )
        print(json.dumps(info, indent=2))
        data.save(str(path_out))
        retrieve_success.append(info["retrieve_success"])
        progress.set_postfix(retrieve=sum(retrieve_success) / len(retrieve_success))


"""
p evaluation.py main
p evaluation.py main --retriever_name bm25_page
p evaluation.py main --output_dir outputs/dummy_eval
p evaluation.py main --output_dir outputs/dummy_eval --retriever_name bm25_page
"""


if __name__ == "__main__":
    Fire()
