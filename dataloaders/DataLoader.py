import os
import pandas as pd
from torch.utils.data import Dataset
from math import ceil
from PIL import Image
import torch

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None, train=False, test=False, val=False, batch_size=64,shuffle=False,random_state=5):
        self.img_dir = img_dir
        self.random_state=random_state
        self.transform = transform
        self.target_transform = target_transform
        self.train_path_list=[]
        self.train_labels=[]
        self.test_path_list=[]
        self.test_labels=[]
        self.batch_size=batch_size
        self.train=train
        self.test=test
        self.val=val
        self.img_dir=img_dir
        self.shuffle=shuffle

        self.createiter()



    def __len__(self):
        return len(self.all_images)
    
    def createiter(self):
        i=0      
        if(self.train==True):
            for top, _, files in os.walk(os.path.join(self.img_dir,'train')):
                if(files!=[]):
                    for j in files:
                        self.train_path_list.append(os.path.join(top,j))
                        self.train_labels.append(i)
                    i+=1
            self.imgs=pd.DataFrame.from_dict({'Address':self.train_path_list,'label':self.train_labels})
        elif(self.test==True):
            for top, _, files in os.walk(os.path.join(self.img_dir,'test')):
                if(files!=[]):
                    for j in files:
                        self.test_path_list.append(os.path.join(top,j))
                        self.test_labels.append(i)
                    i+=1
            self.imgs=pd.DataFrame.from_dict({'Address':self.test_path_list,'label':self.test_labels})
        if(self.shuffle):
            self.imgs=self.imgs.sample(frac=1,random_state=self.random_state)
        if(self.val==True):
            self.imgs = self.imgs.tail(int(len(self.imgs)*0.2))
        else:
            self.imgs = self.imgs.head(int(len(self.imgs)*0.8)) #changed
        self.all_images=[self.imgs.iloc[i*self.batch_size:min((i+1)*self.batch_size,len(self.imgs))] for i in range(ceil(len(self.imgs)/self.batch_size))]
        self.resetiter()

    def resetiter(self):
        self.iter=iter(self.all_images)


    def __getitem__(self):
        imageslabels=next(self.iter)
        img_path = list(imageslabels.values)
        images = []
        labels = []
        for i in range(len(img_path)):
            images.append(self.transform(Image.open(img_path[i][0])))
            if(self.target_transform):
                labels.append(self.target_transform(img_path[i][1]))
            else:    
                labels.append(img_path[i][1])
        return torch.stack(images),torch.Tensor(labels)

    

if __name__=="__main__":
    image=CustomImageDataset("./datasets/cifar10/",train=True,test=False,val=False,batch_size=64,shuffle=True)
    print(type(image.__getitem__()[0][0]))
    print(image.__len__())