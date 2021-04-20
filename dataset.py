import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os,glob, cv2
from PIL import Image
import numpy as np


class MyDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.transform = transform
        self.image_list= []
        names=os.listdir(dir_path)
        for file_name in names:
            if file_name.endswith(".jpg"):
                self.image_list.append(os.path.join(dir_path,file_name) )

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.image_list)

    def __getitem__(self, idx):
        img=Image.open(self.image_list[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        image=[t.numpy() for t in img] 
        image=np.array(image)
        return image

if __name__ == '__main__':
    train_dataset = MyDataset("/git/segment_images")
    num_samples=100
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=(num_samples is None),sampler=num_samples)
    for index, batch_data in enumerate(dataloader):
        print(index, batch_dat.shape)