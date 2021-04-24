import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os,glob, cv2
from PIL import Image
import numpy as np

class kangqiang(Dataset):
    def __init__(self, root, split='train', show=False, transform=None,labels=None):
        train_list_file = "/git/assets/kangqiang_dataset_train.txt"
        self.image_list= []
        self.transform = transform
        self.split = split  # train/test/unlabeled set
        self.root=root
        with open(train_list_file,"r") as filehandle:
            for file_name in filehandle:
                file_name=file_name[:-1]
                if file_name.endswith(".jpg"):
                    self.image_list.append(file_name)
        img = Image.open(os.path.join(self.root,self.image_list[0])).convert("RGB")


    def __len__(self):
        return len(self.image_list)
        

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.root,self.image_list[idx])).convert("RGB")
        img = np.transpose(img, (2, 0,1))

        if self.transform is not None:
            img = self.transform(img)
        return img

class kangqiang_emb(Dataset):
    def __init__(self, root, split='train', show=False, transform1=None,transform2=None,embedding=None,labels=None):
        train_list_file = "/git/assets/kangqiang_dataset_train.txt"
        test_list_file ="/git/assets/kangqiang_dataset_test.txt"
        #splits = ('train', 'train+unlabeled', 'unlabeled', 'test', 'train+test')
        self.root=root
        image_train_list= []
        image_test_list=[]
        self.image_list=[]
        self.transform1 = transform1 
        self.transform2 = transform2
        self.labels=labels
        self.split = split  # train/test/unlabeled set
        if embedding is not None:
            self.embedding = np.load(embedding)
        else:
            self.embedding = None
        
        with open(train_list_file,'r') as filehandle:
            for line in filehandle:
                image_train_list.append(line[:-1])

        with open(test_list_file,'r') as filehandle:
            for line in filehandle:
                image_test_list.append(line[:-1])
        if self.split=="train":
            for i in range(len(image_train_list)):
                self.image_list.append({"train_image":image_train_list[i]})
        elif self.split=="train+test":
            for i in range(min(len(image_train_list),len(image_test_list))):
                self.image_list.append({"train_image":image_train_list[i],"test_image":image_test_list[i]})
        print(self.image_list[0]["train_image"])

        
    def __len__(self):
        return len(self.image_list)
    
    def PilImage2Numpy(img):
        image=[t.numpy() for t in img] 
        return np.array(image)


    def __getitem__(self, idx):
        if self.split == 'train':
            img = Image.open(os.path.join(self.root,self.image_list[idx]["train_image"])).convert("RGB")
            img = np.transpose(img, (2, 0,1))

        elif self.split == 'train+test':
            data_train = Image.open(os.path.join(self.root,self.image_list[idx]["train_image"])).convert("RGB")
            data_test = Image.open(os.path.join(self.root,self.image_list[idx]["test_image"])).convert("RGB")
            data_train = np.transpose(data_train, (2, 0,1))
            data_test = np.transpose(data_test, (2, 0,1))
            img = np.concatenate(data_train, data_test)

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
        
        if emb is not None:
            return img_trans1,img_trans2,emb
        else:
            return img_trans1,img_trans2

    
if __name__ == '__main__':
    train_dataset = kangqiang_emb("/git/segment_images")
    num_samples=100
    dataloader = DataLoader(train_dataset, batch_size=16)
    for index, (image1,image2) in enumerate(dataloader):
        print(index, image1.shape)