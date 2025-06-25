import numpy as np
import torch
import pdb
import copy


def visualize(label_temp,new_label):

    # pdb.set_trace()

    if len(label_temp.shape)==4:
        label_temp = torch.squeeze(label_temp,dim=0)

    label = copy.deepcopy(label_temp).cpu().numpy()
    # new_label = [63,126,189,252]

    combined_label = np.zeros((label.shape[1],label.shape[2]))

    for i in range(len(new_label)):

        loc = np.where(label[i]==1)

        if len(loc[0])>0:
            # pdb.set_trace()
            combined_label[loc] = new_label[i]

    return combined_label

def add_background_label(label):

    if len(label.shape)==3:
        label = torch.unsqueeze(label,dim=0)

    # pdb.set_trace()
    bg = 1 - torch.unsqueeze(torch.sum(label,dim=1),dim=1)
    updated_label = torch.cat((label,bg),dim=1)
    # updated_label = torch.argmax(updated_label,dim=1)


    return updated_label


def get_one_hot_encoding(label,n_class):

    # n_class = 5

    if len(label.shape)==3:
        label = torch.unsqueeze(label,dim=0)

    out_softmax = torch.nn.functional.softmax(label,dim=1)
    pred = torch.argmax(out_softmax,dim=1)
    pred_one_hot_encoding = torch.stack([pred==cl_id for cl_id in range(n_class)], dim=1)*1.0

    return pred_one_hot_encoding


def bias_correction(img,sub_val, lower_limit, upper_limit):
    
    img2 = img - sub_val
    img3 = image_thresholding(img2,lower_limit, upper_limit)
    
    return img3
    

def image_thresholding(img, lower_limit, upper_limit):
    
    img[np.where(img<lower_limit)] = 0
    img[np.where(img>upper_limit)] = 0
    
    return img


def single_to_three_channel(img):

    temp = np.expand_dims(img,axis=0)
    temp2 = np.concatenate((temp,temp,temp),axis=0)

    return temp2


def centercrop(img,size):

    h = img.shape[0]
    c = h//2

    new_img = img[c-size//2:c+size//2,c-size//2:c+size//2]

    return new_img

def normalize(img):

    norm_img = (img - np.mean(img))/(np.std(img)+1e-20)

    return norm_img


def normalize_3D(img):

    norm_stack = []

    for i in range(img.shape[0]):
        norm_img = (img[i] - np.mean(img[i]))/(np.std(img[i])+1e-20)
        norm_stack.append(norm_img)

    return np.array(norm_img)


def get_box(label,ext):

    label = label.cpu().numpy()

    idx = np.where(label==1.0)

    if len(idx[0])>0:

        x_min = np.min(idx[0])
        x_max = np.max(idx[0])
        y_min = np.min(idx[1])
        y_max = np.max(idx[1])

        h = y_max  - y_min
        w = x_max - x_min


        x_min = np.clip(x_min - int(0.5*w*ext),0,255)
        x_max = np.clip(x_max + int(0.5*w*ext),0,255)

        y_min = np.clip(y_min - int(0.5*h*ext),0,255)
        y_max = np.clip(y_max + int(0.5*h*ext),0,255)

        
        box = torch.zeros((256,256)).type(torch.int8)
        box[x_min:x_max,y_min:y_max] = 1.0
    else:
        box = torch.zeros((256,256)).type(torch.int8)



    return box
