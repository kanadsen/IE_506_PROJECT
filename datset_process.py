import os
import os.path as osp
import pickle
import csv
import collections

import numpy as np
import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, Resize


# Defining the dataset
class DataSet:
    '''
     This method initializes the object's state when an instance of the class is created. 
     The method reads a CSV file named "filename: which can be test or train" located in the ./data directory and extracts the image file names and their corresponding labels from it.
     It then creates a list of image file paths and a list of corresponding labels. Each label is an integer that represents the class to which the image belongs. 
     The labels are generated based on the unique WordNet IDs (wnids) present in the CSV file. Finally, the method stores the list of image file paths and labels as instance variables named data and label, respectively. 
     It also stores a list of wnids in the dataset as an instance variable named wnids.
    '''
    def __init__(self,image_size,file_name,data_root):
        self.img_size = image_size
        csv_path = f"{data_root}{file_name}.csv"
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]
        data = []
        label = [] # label for the imgaes to be stored here
        lb = -1
        self.wnids = [] # The labels are given in wnids format
        for l in lines:
            name, wnid = l.split(',') # separating the image names 
            path = osp.join(data_root, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1 # Count the number of labels/classes
            data.append(path)
            label.append(lb)
        self.data = data
        self.label = label
        # Transform the image to standard specifications
        self.transform = transforms.Compose([
                                            transforms.RandomResizedCrop((image_size, image_size)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                        ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        path, label = self.data[index], self.label[index]
        image = self.transform(Image.open(path).convert('RGB'))
        return image # Here we are basically storing the path of the image which will be later used to load the image
    

class EpisodeSampler():
    def __init__(self, label, n_episodes, n_way, n_shot, n_query, n_unlabel):
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.n_per = n_shot + n_query + n_unlabel

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1) # Get indices of the images of label==i
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind) # A 2 D array consisting of array of indices of images containing for all labels

    def __len__(self):
        return self.n_episodes
    
    def __iter__(self):
        for i_batch in range(self.n_episodes):
            batch = []
            classes = torch.randperm(len(self.m_ind))
            select_classes = classes[:self.n_way]
            for c in select_classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                cls_batch = l[pos]
                batch.append(cls_batch)
            batch = torch.stack(batch).t().reshape(-1)
            targets = torch.arange(self.n_way).repeat(self.n_per).long()
            yield batch

filenameToPILImage = lambda x: Image.open(x).convert('RGB')

def loadSplit(splitFile):
            dictLabels = {}
            with open(splitFile) as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)
                for i,row in enumerate(csvreader):
                    filename = row[0]
                    label = row[1]
                    if label in dictLabels.keys():
                        dictLabels[label].append(filename)
                    else:
                        dictLabels[label] = [filename]
            return dictLabels


class EmbeddingDataset(Dataset):

    def __init__(self, dataroot, img_size, type = 'train'):
        self.img_size = img_size
        # Transformations to the image
        if type=='train':
            self.transform = transforms.Compose([filenameToPILImage,
                                                transforms.Resize((img_size, img_size)),
                                                transforms.RandomCrop(img_size, padding=8),
                                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                                ])
        else:
            self.transform = transforms.Compose([filenameToPILImage,
                                                transforms.Resize((img_size, img_size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])

        
        self.ImagesDir = os.path.join(dataroot,'images')
        self.data = loadSplit(splitFile = os.path.join(dataroot,'train' + '.csv'))

        self.data = collections.OrderedDict(sorted(self.data.items()))
        keys = list(self.data.keys())
        self.classes_dict = {keys[i]:i  for i in range(len(keys))} # map NLabel to id(0-99)

        self.Files = []
        self.belong = []

        for c in range(len(keys)):
            num = 0
            num_train = int(len(self.data[keys[c]]) * 9 / 10)
            for file in self.data[keys[c]]:
                if type == 'train' and num <= num_train:
                    self.Files.append(file)
                    self.belong.append(c)
                elif type=='val' and num>num_train:
                    self.Files.append(file)
                    self.belong.append(c)
                num = num+1


        self.__size = len(self.Files)

    def __getitem__(self, index):

        c = self.belong[index]
        File = self.Files[index]

        path = os.path.join(self.ImagesDir,str(File))
        try:
            images = self.transform(path)
        except RuntimeError:
            import pdb;pdb.set_trace()
        return images,c

    def __len__(self):
        return self._size