import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os,glob, cv2
from PIL import Image
import numpy as np


class kangqiang(Dataset):
    def __init__(self, root, split='train', show=False, transform=None):
        self.image_list= []
        names=os.listdir(root)
        self.transform = transform
        self.split = split  # train/test/unlabeled set
        for file_name in names:
            if file_name.endswith(".jpg"):
                self.image_list.append(os.path.join(root,file_name) )

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.image_list)

    def __getitem__(self, idx):
        img=Image.open(self.image_list[idx]).convert("RGB")

        if self.transform is not None:
                    img = self.transform(img)
     
        image=[t.numpy() for t in img] 
        image=np.array(image)
        return image

class kangqiang_emb(Dataset):
    def __init__(self, root, split='train', show=False, transform1=None,transform2=None,embedding=None):
        self.image_list= []
        names=os.listdir(root)
        self.transform1 = transform1
        self.transform2 = transform2
        self.split = split  # train/test/unlabeled set
        if embedding is not None:
            self.embedding = np.load(embedding)
        else:
            self.embedding = None
        for file_name in names:
            if file_name.endswith(".jpg"):
                self.image_list.append(os.path.join(root,file_name) )

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.image_list)

    def __getitem__(self, idx):
        img=Image.open(self.image_list[idx]).convert("RGB")

        """if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

            # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))"""


        if self.embedding is not None:
            emb = self.embedding[idx]
        else:
            emb = None

        if self.transform1 is not None:
                    img_trans1 = self.transform1(img)
        else:
            img_trans1 = img

        if self.transform2 is not None:
            img_trans2 = self.transform2(img)
        else:
            img_trans2 = img
        img_trans1=np.array(img_trans1)
        img_trans2=np.array(img_trans2)
        if emb is not None:
            return img_trans1,img_trans2,emb
        else:
            return img_trans1,img_trans2

if __name__ == '__main__':
    train_dataset = kangqiangDataset("/git/segment_images")
    num_samples=100
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=(num_samples is None),sampler=num_samples)
    for index, batch_data in enumerate(dataloader):
        print(index, batch_dat.shape)