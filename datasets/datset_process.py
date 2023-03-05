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
        csv_path = osp.join('./data', file_name + '.csv')
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
        path, label = np.asarray(self.data)[index], np.asarray(self.label)[index]
        # paths = path.asnumpy()
        image =1  # Dummy image is being returned. return actual image 
        return image, label, path # Here we are basically storing the path of the image which will be later used to load the image