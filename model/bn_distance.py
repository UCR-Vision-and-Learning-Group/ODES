import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils import data
import torch
import torch.optim as optim
from tqdm import tqdm 
from torchmetrics import Accuracy
from torch.optim.lr_scheduler import StepLR
import timm
import os
import torchshow as ts
import pdb
import numpy as np
from model.deeplab_v3 import seg_model
from dataloader.data_loader_stack import *

SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# model = timm.create_model('resnet18', pretrained= False, num_classes=65).to(device)

model = seg_model(num_class=2)
model.load_state_dict(torch.load('/home/eegrad/mdislam/Medical_imaging_segmentation/UDA_with_one_stack/BMC_RUNMC/checkpoint/prostrate_deeplabv3_BCE.pt'))



def get_feature_stat_auto(model):
    mean = []
    var = []
    for name,child in model.named_children():
        for name2, layer in child.named_modules():
            if isinstance(layer, nn.BatchNorm2d):
                mean.append(layer.running_mean.clone())
                var.append(layer.running_var.clone())      
    return mean, var

def configure_model_train(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.eval()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
                m.train()
                m.requires_grad_(True)
                m.track_running_stats = True
                pdb.set_trace()
                m.running_mean = torch.zeros([m.num_features]).cuda()
                m.running_var = torch.zeros([m.num_features]).cuda()
                m.momentum = 1    
    return model


def configure_model_tent(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


#saved stats have the pre_trained mean and var
pre_mean, pre_var = get_feature_stat_auto(model)

test_data = CT_Scan('/home/eegrad/mdislam/Dataset/Processed/PROSTRATE/RUNMC_target/normal_split', target_label = [1], split='train')
test_generator  = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)
    
# test_generator = get_test_generator(test_domain = dataset_name)

for i, sample in enumerate(test_generator):
    img = sample[0].to(device)
    img = torch.squeeze(img,dim=0)
    img = torch.unsqueeze(img,dim=1)
    img_stack = torch.cat((img,img,img),dim=1)

    label = sample[1].to(device)
    
    #make running mean, var = 0, to calculate current mean and var
    
    
    #pass data through model to get curr mean and var
    for j in range(img_stack.shape[0]):

        img = img_stack[j]
        img = torch.unsqueeze(img,dim=0)
        model = configure_model_train(model)
        _ = model(img)
    
        #curr_stats have the current mean and var
        curr_mean, curr_var = get_feature_stat_auto(model)
        pdb.set_trace()
        
        #take model to tent state, running mean, var = None
        # model = configure_model_tent(model)

        # dists = []

        #any dists function can be calculated here: L1/L2/KL DIV
        # dists.append(dist_function(curr_mean[-9:], curr_var[-9:], pre_mean[-9:], pre_var[-9:]).detach().cpu().tolist())
        # print(dists)