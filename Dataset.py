from data_helper import clean_text
import pandas as pd
import numpy as np
from PIL import Image

from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from skimage import io
import torchvision

#https://towardsdatascience.com/building-efficient-custom-datasets-in-pytorch-2563b946fd9f
class IG_img_caption_dataset(Dataset):
  def __init__(self, csv_file,prof_type, transform=None):
    self.type = prof_type
    self.annotations = csv_file[csv_file['User'] == self.type]
    self.transform = transform
    
  def __len__(self):
    return len(self.annotations)

  def __getitem__(self,index):
    img_path = self.annotations.iloc[index,2]
    image = io.imread(img_path,plugin='pil')
    #image = image.astype(np.uint8)
    caption_path = self.annotations.iloc[index,3]
    caption = clean_text(open(caption_path,'r',encoding='utf8').read().split())
    sample = {'image':image,'caption':caption}

    if self.transform:
      image = self.transform(image)
    
    return sample