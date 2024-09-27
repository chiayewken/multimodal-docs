set -e

paths="data/questions/test_academic.json data/questions/test_finance.json data/questions/test_product.json"

python evaluation.py test_retriever $paths --retriever_name bm25 --path outputs/retrieve/test/bm25.json
python evaluation.py test_retriever $paths --retriever_name clip --path outputs/retrieve/test/clip.json
python evaluation.py test_retriever $paths --retriever_name bge --path outputs/retrieve/test/bge.json
python evaluation.py test_retriever $paths --retriever_name colpali --path outputs/retrieve/test/colpali.json