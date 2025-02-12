### Setup

```
conda create -n docs python=3.10 setuptools=69.5.1 -y
conda activate docs
pip install -r requirements.txt
conda-pack -n docs -o env.tar.gz --ignore-missing-files --exclude lib/python3.1
```

### Inference Demo

Download testing data and run inference (processing, retrieval, question-answering)

```
python data_loading.py download_pdfs data/test/metadata.csv data/test
python demo.py main
```
