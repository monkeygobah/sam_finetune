import numpy as np
import matplotlib.pyplot as plt
import tifffile
import os
from patchify import patchify  #Only to handle large images
import random
from scipy import ndimage
import utils
from datasets import Dataset
from PIL import Image
from transformers import SamProcessor, SamModel
from torch.utils.data import DataLoader
import trainer
import torch

if __name__ == '__main__':
  
  image_dir = "cfd_crop_images"
  mask_dir = "cfd_annotations_mask"
  csv = '2024_CFD_key_SIZED.csv'

  # mode_width, mode_height = utils.get_modes(image_dir)
  resized_images,resized_masks, adjusted_boxes, names = utils.resize_image_and_boxes(csv, image_dir, mask_dir)

  # large_images, large_masks = utils.resize_data(image_dir, mask_dir)
  print(resized_images.shape)
  print(resized_masks.shape)

  dataset_dict = {
      "image": [Image.fromarray(img) for img in resized_images],
      "label": [Image.fromarray(mask) for mask in resized_masks],
      "boxes": adjusted_boxes,
      "names": names,
  }

  dataset = Dataset.from_dict(dataset_dict)
      
  utils.plot_sample_img(resized_images, dataset)

  # # # Initialize the processor
  processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

  # # # Create an instance of the SAMDataset
  train_dataset = trainer.SAMDataset(dataset=dataset, processor=processor)


  # # # Create a DataLoader instance for the training dataset
  train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)
  
  
  # # # Load the model
  model = SamModel.from_pretrained("facebook/sam-vit-huge")

  # # make sure we only compute gradients for mask decoder
  for name, param in model.named_parameters():
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
      param.requires_grad_(False)
  
  trainer.train(train_dataloader, model)
    

