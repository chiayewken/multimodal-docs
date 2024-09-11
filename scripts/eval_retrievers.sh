set -e

python evaluation.py test_retriever data/questions/test.json --retriever_name bm25 --path outputs/retrieve/test/bm25.json
python evaluation.py test_retriever data/questions/test.json --retriever_name clip --path outputs/retrieve/test/clip.json
python evaluation.py test_retriever data/questions/test.json --retriever_name bge --path outputs/retrieve/test/bge.json
python evaluation.py test_retriever data/questions/test.json --retriever_name colpali --path outputs/retrieve/test/colpali.json