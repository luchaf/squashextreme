# adapted from: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb

from transformers import DetrImageProcessor
import torchvision
import os
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import DetrForObjectDetection, TableTransformerForObjectDetection
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import wandb  # Import W&B
from pytorch_lightning.loggers import WandbLogger  # Import the WandbLogger

wandb.init(project="table_transformer")

class TransformersCheckpointCallback(Callback):
    def __init__(self, save_path, monitor='validation_loss', mode='min'):
        self.save_path = save_path
        self.monitor = monitor
        self.mode = mode
        self.best_score = None

    def on_validation_end(self, trainer, pl_module):
        # Get the metric value
        current_score = trainer.callback_metrics.get(self.monitor)

        if current_score is None:
            return

        # Check if this is the best score
        if self.best_score is None or (self.mode == 'min' and current_score < self.best_score) or (self.mode == 'max' and current_score > self.best_score):
            self.best_score = current_score
            # Save the model using save_pretrained
            pl_module.model.save_pretrained(self.save_path)
        
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, "train.json" if train else "val.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):       
        # # read in PIL image and target in COCO format
        # # feel free to add data augmentation here before passing them to the next step
        img, target = super(CocoDetection, self).__getitem__(idx)

        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = processor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch

class Detr(pl.LightningModule):
     def __init__(self, lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4):
         super().__init__()
         # replace COCO classification head with custom head
         # we specify the "no_timm" variant here to not rely on the timm library
         # for the convolutional backbone
         self.model = TableTransformerForObjectDetection.from_pretrained(
      "microsoft/table-transformer-structure-recognition",
      #"luchaf/testest2",
      ignore_mismatched_sizes=True,
  )

         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay

     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs

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
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

        return optimizer

     def train_dataloader(self):
        return train_dataloader

     def val_dataloader(self):
        return val_dataloader

processor = DetrImageProcessor()
train_dataset = CocoDetection(img_folder='data/images/table_structure_recognition_modeling_data/train', processor=processor)
val_dataset = CocoDetection(img_folder='data/images/table_structure_recognition_modeling_data/val', processor=processor, train=False)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))

train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=1)
batch = next(iter(train_dataloader))

model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

# Initialize WandbLogger
wandb_logger = WandbLogger()

# Callback to save the best model based on validation loss

checkpoint_callback = TransformersCheckpointCallback(
    save_path='tatr_model/best_model',
    monitor='validation_loss',
    mode='min'
)

trainer = Trainer(
    max_steps=500,
    gradient_clip_val=0.1,
    callbacks=[checkpoint_callback],
    logger=wandb_logger, 
)

trainer.fit(model)

wandb.finish()
