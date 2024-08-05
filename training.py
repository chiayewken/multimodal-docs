from functools import partial
from typing import Tuple, List, Union

import torch
from PIL import Image
from datasets import Dataset
from fire import Fire
from pydantic import BaseModel
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, Trainer
from transformers import TrainingArguments

from data_loading import convert_image_to_text, convert_text_to_image

TrainingSample = Tuple[List[Union[str, Image.Image]], str]


def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)
    return image.resize((new_width, new_height))


class ModelTrainer(BaseModel):
    save_dir: str
    epochs: int
    base_model: str

    def run(self, samples: List[TrainingSample]) -> None:
        pass


class GemmaTrainer(ModelTrainer):
    base_model: str = "google/paligemma-3b-pt-896"
    max_image_size: int = 896

    @staticmethod
    def collate_fn(examples, processor, device):
        assert len(examples) == 1
        tokens = processor(
            text=examples[0]["text"] or None,
            images=[convert_text_to_image(x) for x in examples[0]["images"]] or None,
            suffix=examples[0]["labels"],
            return_tensors="pt",
            padding="longest",
        )

        tokens = tokens.to(torch.bfloat16).to(device)
        return tokens

    def run(self, samples: List[TrainingSample]) -> None:
        data = []
        for inputs, targets in samples:
            data.append(
                dict(
                    text="\n\n".join([x for x in inputs if isinstance(x, str)]),
                    images=[
                        convert_image_to_text(resize_image(x, self.max_image_size))
                        for x in inputs
                        if isinstance(x, Image.Image)
                    ],
                    labels=targets,
                )
            )

        processor = PaliGemmaProcessor.from_pretrained(self.base_model)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.base_model, torch_dtype=torch.bfloat16
        ).cuda()

        for param in model.vision_tower.parameters():
            param.requires_grad = True
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = True

        args = TrainingArguments(
            num_train_epochs=self.epochs,
            remove_unused_columns=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=64,
            warmup_ratio=0.05,
            learning_rate=2e-5,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=100,
            save_strategy="no",
            bf16=True,
            output_dir=self.save_dir,
            overwrite_output_dir=True,
            label_smoothing_factor=0.1,
            dataloader_pin_memory=False,  # Save memory
            gradient_checkpointing=True,  # Save memory
        )

        trainer = Trainer(
            model=model,
            train_dataset=Dataset.from_list(data),
            data_collator=partial(self.collate_fn, processor=processor, device="cuda"),
            args=args,
        )
        trainer.train()
        trainer.save_model()


if __name__ == "__main__":
    Fire()
