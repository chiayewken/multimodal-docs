"""
Simple demo for multimodal document retrieval and question answering with PDF files.
"""
from fire import Fire

from data_loading import MultimodalDocument, YoloDetector, MultimodalSample
from modeling import select_model
from retrieval import select_retriever


def main(
    path: str = "data/test/NYSE_FBHS_2023.pdf",
    query: str = "Can you explain the stock price trend based on the graph?",
    retriever_name: str = "colpali",
    generator_name: str = "qwen",
    top_k: int = 5,
):
    print(locals())
    detector = YoloDetector()
    doc = MultimodalDocument.load_from_pdf(path, detector=detector)
    sample = MultimodalSample(question=query, answer="", category="")

    retriever = select_retriever(retriever_name)
    output = retriever.run(sample.question, doc)
    sorted_pages = sorted(output.pages, key=lambda p: p.score, reverse=True)
    sample.retrieved_pages = sorted([p.number for p in sorted_pages][:top_k])

    context = []
    for p in doc.pages:
        if p.number in sample.retrieved_pages:
            if p.text:
                context.append(p.text)
            context.extend(o.get_image() for o in p.get_tables_and_figures())

    generator = select_model(generator_name)
    inputs = [
        "Context:",
        *context,
        f"Answer the following question in 200 words or less: {sample.question}",
    ]
    sample.pred = generator.run(inputs)
    sample.generator = generator_name
    print(sample.model_dump_json(indent=2))


"""
p demo.py main
"""

if __name__ == "__main__":
    Fire()
