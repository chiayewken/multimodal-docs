### Setup

```
conda create -n docs python=3.10 -y
conda activate docs
pip install -r requirements.txt
```

### Inference Demo

Run model prediction on several questions with GPT-4V for the whole AMLX financial report (texts, tables, figures).

```
cd data/demo
unzip NASDAQ_AMLX_2022.zip
cd ../..
python demo.py main --generator_name openai_vision --data_dir data/demo/NASDAQ_AMLX_2022 --top_k 10 --output_dir outputs/demo/amlx
```

### API Setup

[Gemini Pro](https://ai.google.dev/tutorials/python_quickstart?hl=en) (multimodal): Please create a file
named `gemini_vision_info.json`

```
{"engine": "gemini-pro-vision", "key": "your_api_key"}
```

[GPT-4(V)](https://platform.openai.com/docs/guides/vision) (multimodal): Please create a file
named `openai_vision_info.json`

```
{"engine": "gpt-4-vision-preview", "key": "your_api_key"}
```
