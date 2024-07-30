from functools import partial

from datasets import load_dataset
from fire import Fire
from transformers import TrainingArguments

from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, Trainer

import torch


def collate_fn(examples, processor, device):
    texts = ["answer " + example["question"] for example in examples]
    labels = [example["multiple_choice_answer"] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    tokens = processor(
        text=texts, images=images, suffix=labels, return_tensors="pt", padding="longest"
    )

    # for key, value in tokens.items():
    #     print(dict(key=key, value=value.shape))
    tokens = tokens.to(torch.bfloat16).to(device)
    return tokens


def train_gemma(
    data_path="HuggingFaceM4/VQAv2",
    model_id="google/paligemma-3b-mix-448",
    device="cuda",
):
    # https://huggingface.co/blog/paligemma
    ds = load_dataset(data_path, split="train")
    cols_remove = ["question_type", "answers", "answer_type", "image_id", "question_id"]
    ds = ds.remove_columns(cols_remove)
    ds = ds.train_test_split(test_size=0.1)
    train_ds = ds["train"]
    val_ds = ds["test"]

    processor = PaliGemmaProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(device)

    for param in model.vision_tower.parameters():
        param.requires_grad = True

    for param in model.multi_modal_projector.parameters():
        param.requires_grad = True

    args = TrainingArguments(
        num_train_epochs=2,
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64,
        warmup_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        adam_beta2=0.999,
        logging_steps=100,
        optim="adamw_hf",
        save_strategy="epoch",
        bf16=True,
        dataloader_pin_memory=False,
        output_dir="train_outputs",
        overwrite_output_dir=True,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=partial(collate_fn, processor=processor, device=device),
        args=args,
    )
    trainer.train()


if __name__ == "__main__":
    Fire()
