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
        names=os.listdir(root)
        with open(train_list_file,"r") as filehandle:
            for file_name in filehandle:
                if file_name.endswith(".jpg"):
                    self.image_list.append(file_name)

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.image_list)
    
    def __loadfile(self, data_file):
        print(os.path.join(self.root, data_file))
        path_to_data = os.path.join(self.root, data_file)
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))
        return images
        

    def __getitem__(self, idx):
        self.data = self.__loadfile(self.image_list[idx])
        if self.labels is not None:
            img, target = self.data[idx], int(self.labels[idx])
        else:
            img, target = self.data[idx], None
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)
        return img

class kangqiang_emb(Dataset):
    def __init__(self, root, split='train', show=False, transform1=None,transform2=None,embedding=None):
        train_list_file = "/git/assets/kangqiang_dataset_train.txt"
        test_list_file ="/git/assets/kangqiang_dataset_test.txt"
        #splits = ('train', 'train+unlabeled', 'unlabeled', 'test', 'train+test')
        self.root=root
        self.image_train_list= []
        self.image_test_list=[]
        names=os.listdir(root)
        self.transform1 = transform1 
        self.transform2 = transform2
        self.split = split  # train/test/unlabeled set
        if embedding is not None:
            self.embedding = np.load(embedding)
        else:
            self.embedding = None
        
        with open(train_list_file,'r') as filehandle:
            for line in filehandle:
                self.image_train_list.append(line.splitlines())
            filehandle.close()

        with open(test_list_file,'r') as filehandle:
            for line in filehandle:
                self.image_test_list.append(line.splitlines())
            filehandle.close()


        """for file_name in names:
            if file_name.endswith(".jpg"):
                self.image_list.append(os.path.join(root,file_name))"""

    def __len__(self):
        # 返回数据集的数据数量
        return len(self.image_train_list),len(self.image_test_list)

  
    def __loadfile(self, data_file):
        path_to_data = os.path.join(self.root, data_file)
        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))
        return images

    def __getitem__(self, idx):
        """
        img=Image.open(self.image_list[idx]).convert("RGB")
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

            # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        """
      
        
        if self.split == 'train':
            self.data = self.__loadfile(self.image_train_list[idx])
        elif self.split == 'train+test':
            data_train = self.__loadfile(self.image_train_list[idx])
            data_test = self.__loadfile(self.image_test_list[idx])
            self.data = np.concatenate(data_train, data_test)
        
        if self.labels is not None:
            img, target = self.data[idx], int(self.labels[idx])
        else:
            img, target = self.data[idx], None
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))


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
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=(num_samples is None),sampler=num_samples)
    for index, batch_data in enumerate(dataloader):
        print(index, batch_dat.shape)
