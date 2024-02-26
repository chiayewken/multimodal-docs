### Setup

```
conda create -n finance python=3.10 -y
conda activate finance
pip install -r requirements.txt
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
