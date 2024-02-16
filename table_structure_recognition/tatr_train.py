import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision
import pytorch_lightning as pl
from transformers import DetrForObjectDetection, TableTransformerForObjectDetection, DetrImageProcessor
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb
import argparse

class TransformersCheckpointCallback(Callback):
    def __init__(self, save_path, monitor='validation_loss', mode='min'):
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.best_score = None

    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return
        if self.best_score is None or (self.mode == 'min' and current_score < self.best_score) or (self.mode == 'max' and current_score > self.best_score):
            self.best_score = current_score
            pl_module.model.save_pretrained(self.save_path)

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, "train.json" if train else "val.json")
        super().__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return pixel_values, target

class CollateFunction:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        pixel_values = [item[0] for item in batch]
        encoding = self.processor.pad(pixel_values, return_tensors="pt")
        labels = [item[1] for item in batch]
        batched = {'pixel_values': encoding['pixel_values'], 'pixel_mask': encoding['pixel_mask'], 'labels': labels}
        return batched

class Detr(pl.LightningModule):
    def __init__(self, model_name, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4):
        super().__init__()
        self.model = TableTransformerForObjectDetection.from_pretrained(
            model_name, ignore_mismatched_sizes=True,
        )
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss
        loss_dict = outputs.loss_dict
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())
        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad], "lr": self.lr_backbone},
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

def main(args):
    # Initialize WandbLogger with dynamic project name based on the task
    wandb_logger = WandbLogger(project=f"table_transformer_{args.task}", log_model=False)

    os.makedirs(args.save_path, exist_ok=True)

    processor = DetrImageProcessor()

    # Instantiate the collate function class with the processor
    collate_fn_instance = CollateFunction(processor)

    model_name = {
        'recognition': "microsoft/table-transformer-structure-recognition",
        'detection': "microsoft/table-transformer-detection"
    }[args.task]

    train_dataset = CocoDetection(img_folder=f'{args.data_path}/train', processor=processor)
    val_dataset = CocoDetection(img_folder=f'{args.data_path}/val', processor=processor, train=False)

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_instance, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn_instance, batch_size=args.batch_size)

    model = Detr(model_name, args.lr, args.lr_backbone, args.weight_decay)

    checkpoint_callback = TransformersCheckpointCallback(save_path=args.save_path, monitor='validation_loss', mode='min')
    
    early_stopping_callback = EarlyStopping(
        monitor='validation_loss',
        mode='min',
        patience=15,
        verbose=True,
        min_delta=0.00
    )    
    
    trainer = Trainer(max_steps=args.max_steps, gradient_clip_val=args.gradient_clip_val, callbacks=[checkpoint_callback, early_stopping_callback], logger=wandb_logger)
    trainer.fit(model, train_dataloader, val_dataloader)

    wandb.finish()

# python tatr_train.py --task detection --data_path data/images/table_detection_modeling_data --save_path tatr_model/table_detection/best_model
# python tatr_train.py --task recognition --data_path data/images/table_structure_recognition_modeling_data --save_path tatr_model/table_recognition/best_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a table transformer model for recognition or detection.")
    parser.add_argument("--task", choices=["recognition", "detection"], required=True, help="Task to train for: 'recognition' or 'detection'.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--save_path", type=str, default="best_model", help="Path to save the best model.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for the backbone.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--max_steps", type=int, default=10000, help="Max training steps.")
    parser.add_argument("--gradient_clip_val", type=float, default=0.1, help="Gradient clipping value.")

    args = parser.parse_args()

    main(args)
