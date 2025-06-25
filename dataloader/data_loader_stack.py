import os
import torch

import torch.nn.functional  as F
from torchvision import datasets, transforms
import cv2, tifffile
import numpy as np
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import numpy as np
import tifffile
import copy
# from utils import *
import pdb
import nibabel as nib



# def read_data(img_file):


#     if img_file.endswith('.tif'):

#         data = tifffile.imread(img_file)

#     elif img_file.endswith('.nii.gz'):
#         img = nib.load(img_file)
#         data = img.get_fdata()

#     return data


def read_data(img_file):
    """Read image data from .tif, .nii.gz, or .npy files."""
    if img_file.endswith('.tif'):
        return tifffile.imread(img_file)
    elif img_file.endswith('.nii.gz'):
        return nib.load(img_file).get_fdata()
    elif img_file.endswith('.npy'):
        return np.load(img_file)
    else:
        raise ValueError(f"Unsupported file format: {img_file}")


def normalize(img):

    norm_img = (img - np.mean(img))/(np.std(img)+1e-20)
    norm_img = (norm_img-np.min(norm_img))/(np.max(norm_img)-np.min(norm_img)+1e-20)

    return norm_img


def normalize_3D(img):

    norm_stack = []

    for i in range(img.shape[0]):
        norm_stack.append(normalize(img[i]))

    return np.array(norm_stack)

class Stack_Loader(Dataset):
    
    def __init__(self,data_dir,target_label, split='train', file_idx = None, augmentation = 0):


        
        
        self.data_dir = data_dir
        self.split = split
        self.target_label = target_label

        self.augmentation_flag = augmentation

        self.img_dir = os.path.join(self.data_dir,self.split,'data')
        self.label_dir = os.path.join(self.data_dir,self.split,'labels')


        self.all_img_files =  sorted([f for f in os.listdir(self.img_dir) if f.endswith('.tif')])
        self.all_label_files =  sorted([f for f in os.listdir(self.label_dir) if f.endswith('.tif')])

        if file_idx is None:
            
            self.img_files = self.all_img_files
            self.label_files = self.all_label_files

        else:
            self.img_files = [self.all_img_files[f] for f in file_idx]
            self.label_files = [self.all_label_files[f] for f in file_idx]

        
        self.resize_height = 256
        self.resize_width = 256
        
        
        self.img_files =[os.path.join(self.img_dir,self.img_files[stack_id]) for stack_id in range(len(self.img_files)) ]
        self.label_files =[os.path.join(self.label_dir,self.label_files[stack_id]) for stack_id in range(len(self.label_files)) ]

        print(len(self.img_files) , len(self.label_files))
        assert len(self.img_files) == len(self.label_files)


    def augmentation(self,image,label):
        
        image = torch.unsqueeze(image,0)
        label = torch.unsqueeze(label,0)
        temp = torch.concat([image,label], 0)


        transform1 = transforms.Compose([transforms.RandomVerticalFlip(p=1.0)])
        transform2 = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)])
        transform3 = transforms.Compose([transforms.RandomRotation(degrees=(-30, 30),center=(128, 128))])
        transform4 = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomRotation(degrees=(-30, 30),center=(128, 128))])
        transform5 = transforms.Compose([transforms.RandomVerticalFlip(p=1.0), transforms.RandomRotation(degrees=(-30, 30),center=(128, 128))])     


        n = np.random.randint(6)

        if n==0:
            temp2 = temp
        if n==1:
            temp2 = transform1(temp)
        if n==2:
            temp2 = transform2(temp)
        if n==3:
            temp2 = transform3(temp)
        if n==4:
            temp2 = transform4(temp)
        if n==5:
            temp2 = transform5(temp)

        return temp2[0],temp2[1]


        
    def __len__(self):
        if self.augmentation_flag==1:
            return 4*len(self.img_files)
        else:
            # return self.added_len[-1]
            return len(self.img_files)



    def __getitem__(self, idx):

        stack_id = idx % len(self.img_files)

        img = read_data(self.img_files[stack_id])
        label = read_data(self.label_files[stack_id])

        # pdb.set_trace()

        one_hot_label = np.zeros((label.shape[0],len(self.target_label),label.shape[1],label.shape[2]))

        for t_l in range(len(self.target_label)):

            mask = np.zeros_like(label)
            loc = np.where(label==self.target_label[t_l])

            if len(loc[0])>0:
                mask[loc] = 1
                one_hot_label[:,t_l,:,:] = mask


        

        img_buffer = torch.from_numpy(img*1.0)
        label_buffer = torch.from_numpy(one_hot_label)


        img_buffer, label_buffer = img_buffer.type(torch.DoubleTensor), label_buffer.type(torch.DoubleTensor)

        if img_buffer.shape[1]!=self.resize_height or img_buffer.shape[2]!=self.resize_width:
            
            img_buffer = torch.unsqueeze(img_buffer,dim=1)

            img_buffer = F.interpolate(img_buffer,size = [self.resize_height,self.resize_width],mode='bilinear') 
            label_buffer = F.interpolate(label_buffer,size = [self.resize_height,self.resize_width],mode='bilinear')

            img_buffer = torch.squeeze(img_buffer,dim=1)


        img_buffer = torch.from_numpy(normalize_3D(img_buffer.numpy()))
        label_buffer = (label_buffer>0)


        if self.augmentation_flag == 1:
            img_buffer,label_buffer = self.augmentation(img_buffer,label_buffer)


        assert len(img_buffer.shape)==3
        assert len(label_buffer.shape)==4


        return img_buffer,label_buffer,self.img_files[stack_id]





if __name__ == "__main__":
    
    # DATA_DIR = '/home/eegrad/mdislam/Dataset/Processed/DUKE/MRI/'
    DATA_DIR = '/home/eegrad/mdislam/Dataset/Processed/CHAOS/MRI/T1DUAL_OutPhase/'

    # test_data = Stack_Loader(DATA_DIR, target_label = [1], split='test',file_idx = None)
    test_data = Stack_Loader(DATA_DIR, target_label = [63,126,189,252], split='test',file_idx = None)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=1)
    
    print(len(test_loader))

    n = test_data.__len__()
    [a,b,c] = test_data.__getitem__(1)
    print(n, a.shape, b.shape,c)
    pdb.set_trace()

    for i, sample in enumerate(test_loader):
        inputs = sample[0]
        gt_labels = sample[1]

        print(inputs.size(),gt_labels.size())
        # print(torch.unique(gt_labels))

        if i == 2:
            break



