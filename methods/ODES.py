import random, copy
from copy import deepcopy
import numpy as np

import torch
import torch.jit
import torch.nn.functional as F
import torch.nn as nn

from active.floating_region import FloatingRegionScore
from active.spatial_purity import SpatialPurity
from loss.negative_learning_loss import NegativeLearningLoss,Pseudo_label_Loss

from utils.helper_functions import *

from conf import cfg, load_cfg_fom_args
import tifffile
import copy
from skimage import morphology
import pdb

from torchvision import datasets, transforms

from FFT_weight_2 import farthest_first_with_reference_distance, get_weight_map, expand_labels_with_cosine_similarity
# , plot_weight_map 

#########################################

# Criterion

# BCE_criterion = torch.nn.BCEWithLogitsLoss()
BCE_criterion = torch.nn.BCELoss()
sup_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
negative_criterion = NegativeLearningLoss(threshold=0.05)
pseudo_label_criterion = Pseudo_label_Loss(threshold=0.8)
# device = torch.device('cuda')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Continuity_loss(pred):

    pred_1 = torch.argmax(pred.softmax(1),dim=1)
    n1 = int(0.2*pred.shape[0])
    l1 = sup_criterion(pred[n1:-n1],pred_1[n1+1:-n1+1])

    return l1


def Pixel_Selection_Equal(model,img,label,slice_selection_info,cfg):

    model.eval()
    with torch.no_grad():
        # pdb.set_trace()
        primary_pred,feature,concat_feature = model(img,require_feature=True)
        
    if cfg.ACTIVE.BUDGET_SLICE<1:
        selected_idx = Slice_Selection_Module(cfg,model,img,feature,slice_selection_info)
    else:
        selected_idx = np.arange(img.shape[0])

    label = torch.argmax(label,dim=1)
    calculate_purity = SpatialPurity(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()

    p = torch.softmax(primary_pred, dim=1)
    entropy = torch.sum(-p * torch.log(p + 1e-6), dim=1)
    pseudo_label = torch.argmax(p, dim=1)
    one_hot = F.one_hot(pseudo_label, num_classes=cfg.MODEL.NUM_CLASSES).float()
    one_hot = one_hot.permute((0,3, 1, 2))
    purity = calculate_purity(one_hot).squeeze(dim=1)
    score_stack = torch.mul(entropy,purity).cpu().detach().numpy()
        

    n_pixel = int(np.round(cfg.ACTIVE.BUDGET_PIXEL*score_stack.shape[1]*score_stack.shape[2]/100))
    tgt_mask = (torch.zeros(score_stack.shape) + 255).to(device)

    for idx in selected_idx:


        score_slice = score_stack[idx]

        mask = torch.zeros(score_slice.shape).to(device)
        sorted_slice_idx = sort_locations_2d(score_slice)
        
        selected_pixel_loc = sorted_slice_idx[-n_pixel:]
        
        mask[selected_pixel_loc[:,0],selected_pixel_loc[:,1]] = 1
        tgt_mask[idx] = (tgt_mask[idx]*(1-mask) + mask*label[idx]) 


    tgt_mask = tgt_mask.type(torch.LongTensor).to(device)
    model.train()

    return tgt_mask


def Slice_Selection_Module(cfg,model,img,feature,slice_selection_info):
    
    dists_from_source,sorted_slice_idx = Distance_Calculation_from_Source(model,img,slice_selection_info)
    dists_from_source = dists_from_source/dists_from_source.max()
    dists_from_source = torch.from_numpy(dists_from_source).to(device)
    # pdb.set_trace()
    selected_indices = farthest_first_with_reference_distance(feature[1],dists_from_source,100*cfg.ACTIVE.BUDGET_SLICE,device)
    return selected_indices


def Region_Selection_Equal(model,img,label,slice_selection_info,cfg):

    model.eval()
    with torch.no_grad():
        # pdb.set_trace()
        primary_pred,feature,concat_feature = model(img,require_feature=True)

    if cfg.ACTIVE.BUDGET_SLICE<1:
        selected_idx = Slice_Selection_Module(cfg,model,img,feature,slice_selection_info)
    else:
        selected_idx = np.arange(img.shape[0])


    # pdb.set_trace()

    label = torch.argmax(label,dim=1)
    floating_region_score = FloatingRegionScore(in_channels=cfg.MODEL.NUM_CLASSES, size=2 * cfg.ACTIVE.RADIUS_K + 1).cuda()
    per_region_pixels = (2 * cfg.ACTIVE.RADIUS_K + 1) ** 2
    active_radius = cfg.ACTIVE.RADIUS_K
    mask_radius = cfg.ACTIVE.RADIUS_K * 2


    score_stack = []

    for i1 in range(primary_pred.shape[0]):
        score_slice, purity, entropy = floating_region_score(torch.unsqueeze(primary_pred[i1],dim=0))
        # score_slice = score_slice.cpu().detach().numpy()
        score_slice = (purity * entropy).cpu().detach().numpy()
        score_stack.append(score_slice)

    score_stack = np.array(score_stack)
    # score_stack = np.clip(np.array(score_stack),0,1000)

    # pdb.set_trace()
    n_region = int(np.round(cfg.ACTIVE.BUDGET_REGION*0.01*score_stack.shape[1]*score_stack.shape[2]/per_region_pixels))
    tgt_mask = (torch.zeros(score_stack.shape) + 255).to(device)

    # pdb.set_trace()

    for idx in selected_idx:

        temp = copy.deepcopy(score_slice)
        score_slice = np.clip(score_slice,0,1000)
        score_slice = score_stack[idx] / np.max(score_stack[idx])

        # score_slice = (score_stack[idx]-np.min(score_stack[idx]))/(np.max(score_stack[idx])-np.min(score_stack[idx]))

        # weights_cos,weights_eu, cosine_distance, euclidean_distance = get_weight_map(heat_map = score_slice,
        #                                                                             threshold = 0.5 ,
        #                                                                             features = feature[0][idx],
        #                                                                             k_percent = cfg.ACTIVE.BUDGET_REGION,
        #                                                                             alpha_cos = 5,
        #                                                                             alpha_eu = 5,
        #                                                                             plot=0)


        weight, cosine_distance, euclidean_distance = get_weight_map(heat_map = score_slice,
                                                                                    threshold = 0.5 ,
                                                                                    features = feature[0][idx],
                                                                                    k_percent = cfg.ACTIVE.BUDGET_REGION,
                                                                                    alpha_cos = 5,
                                                                                    alpha_eu = 5,
                                                                                    plot=0)

        # weight = torch.from_numpy(weights_cos * weights_eu)
        if np.all(weight):
            score_slice = temp

        weight = torch.from_numpy(weight)
        weight_resized = F.interpolate((weight.unsqueeze(0)).unsqueeze(0), size= score_stack[idx].shape, mode='bilinear', align_corners=False).squeeze()
        weight_resized = weight_resized.numpy()

        # score_slice = np.multiply(score_slice,weight_resized)

        mask = torch.zeros(score_slice.shape).to(device)
        

        for pixel in range(n_region):

            val_max = np.max(score_slice)
            loc_max = np.where(score_slice==val_max)
            loc_max = [loc_max[0][0],loc_max[1][0]]

            active_start_x = np.clip(loc_max[0] - active_radius,0,256)
            active_end_x = np.clip(loc_max[0] + active_radius + 1 ,0,256)

            active_start_y = np.clip(loc_max[1] - active_radius,0,256)
            active_end_y = np.clip(loc_max[1] + active_radius + 1 ,0,256)


            mask_start_x = np.clip(loc_max[0] - mask_radius,0,256)
            mask_end_x = np.clip(loc_max[0] + mask_radius + 1 ,0,256)

            mask_start_y = np.clip(loc_max[1] - mask_radius,0,256)
            mask_end_y = np.clip(loc_max[1] + mask_radius + 1 ,0,256)

            mask[active_start_x:active_end_x,active_start_y:active_end_y] = 1
            score_slice[active_start_x:active_end_x,active_start_y:active_end_y] = -float('inf')
            score_slice[mask_start_x:mask_end_x,mask_start_y:mask_end_y] = -float('inf')



 
        # al_slice = (tgt_mask[idx]*(1-mask) + mask*label[idx]) 

        # # pdb.set_trace()
        # primary_label = copy.deepcopy(al_slice)
        # expanded_primary_label = expand_labels_with_cosine_similarity(primary_label, 1-cosine_distance, 0.9)
        # tgt_mask[idx] = expanded_primary_label

        tgt_mask[idx] = tgt_mask[idx]*(1-mask) + mask*label[idx]



    tgt_mask = tgt_mask.type(torch.LongTensor).to(device)
    model.train()

    return tgt_mask

def plot_heat_map(img,name):
    tifffile.imwrite('heatmaps/{0}'.format(name),img)


def sort_locations_3d(array):
    indices = np.argsort(array, axis=None)
    sorted_locations = np.unravel_index(indices, array.shape)

    loc = []
    for i in range(len(sorted_locations[0])):
        x, y, z = sorted_locations[0][i], sorted_locations[1][i], sorted_locations[2][i]
        loc.append(np.array([x,y,z]))

    loc = np.array(loc)

    return loc


def sort_locations_2d(array):
    indices = np.argsort(array, axis=None)
    sorted_locations = np.unravel_index(indices, array.shape)

    loc = []
    for i in range(len(sorted_locations[0])):
        x, y = sorted_locations[0][i], sorted_locations[1][i]
        loc.append(np.array([x,y]))

    loc = np.array(loc)

    return loc


def get_one_hot_encoding(label,n_class):

    if len(label.shape)==3:
        label = torch.unsqueeze(label,dim=0)

    out_softmax = torch.nn.functional.softmax(label,dim=1)
    pred = torch.argmax(out_softmax,dim=1)
    pred_one_hot_encoding = torch.stack([pred==cl_id for cl_id in range(n_class)], dim=1)*1.0

    return pred_one_hot_encoding

def Entropy_loss(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def instance_to_border(instance_mask,background):
   
    # get the membrane mask
    membrane_mask = morphology.dilation(instance_mask, selem=morphology.ball(1)) - instance_mask
    membrane_mask = membrane_mask != background
    membrane_mask = membrane_mask.astype(np.float32)

    return membrane_mask


class TTA_Active(nn.Module):
    """TTA_Active
    """
    def __init__(self, model, optimizer,pre_mean,pre_var,cfg):
        super().__init__()

        self.cfg = cfg

        self.model = model
        self.optimizer = optimizer
        self.steps = cfg.OPTIM.STEPS

        # self.rst = cfg.ACTIVE.RST
        # self.ap = ap
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = cfg.MODEL.EPISODIC
        self.budget_slice = cfg.ACTIVE.BUDGET_SLICE

        self.pre_mean = pre_mean
        self.pre_var = pre_var


        self.stack_no = 0        
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)



    def forward(self, x, gt_label,ac_flag=1):

        if self.episodic:
            self.reset()


        x_new = copy.deepcopy(x)

        for _ in range(self.steps):
            
            if ac_flag==0:
                # print('ac flag: ',ac_flag)
                self.model.eval()
                with torch.no_grad():
                    out = self.model(x_new)
                
            else:
                # print('ac flag: ',ac_flag)
                out = self.forward_and_adapt(x_new,gt_label)


        self.model.train()

        return self.model.state_dict(),out
        # return out


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, gt_label):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        slice_selection_info = [self.pre_mean,self.pre_var]

        self.optimizer.zero_grad()
        if cfg.ACTIVE.MODE == 'PA':
            # print("Pixel")
            tgt_mask = Pixel_Selection_Equal(self.model,x,gt_label,slice_selection_info,cfg)\

        elif cfg.ACTIVE.MODE == 'RA':
            # print("Patch")
            tgt_mask = Region_Selection_Equal(self.model,x,gt_label,slice_selection_info,cfg)


        self.model.train()
        primary_pred = self.model(x)
        loss1 = sup_criterion(primary_pred, tgt_mask)
        loss2 = Continuity_loss(primary_pred)
        loss1.backward()
        self.optimizer.step()
        self.stack_no+=1
        return primary_pred


    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved self.model/optimizer state")
        self.load_model_and_optimizer()


    def load_model_and_optimizer(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    def copy_model_and_optimizer(self,model, optimizer):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(model.state_dict())
        optimizer_state = deepcopy(optimizer.state_dict())
        return model_state, optimizer_state





def augmentation(image,n_aug):
    
    temp = torch.unsqueeze(image,0)

    transform1 = transforms.Compose([transforms.RandomVerticalFlip(p=1.0)])
    transform2 = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)])
    transform3 = transforms.Compose([transforms.RandomRotation(degrees=(-30, 30),center=(128, 128))])
    transform4 = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), transforms.RandomRotation(degrees=(-30, 30),center=(128, 128))])
    transform5 = transforms.Compose([transforms.RandomVerticalFlip(p=1.0), transforms.RandomRotation(degrees=(-30, 30),center=(128, 128))])     

    augmented_image = []
    for i in range(n_aug):

        n = i%6
        

        if n==0:
            temp2 = temp
        elif n==1:
            temp2 = transform1(temp)
        elif n==2:
            temp2 = transform2(temp)
        elif n==3:
            temp2 = transform3(temp)
        elif n==4:
            temp2 = transform4(temp)
        elif n==5:
            temp2 = transform5(temp)

        augmented_image.append(temp2.cpu().numpy())
    
    augmented_image = torch.squeeze(torch.from_numpy(np.array(augmented_image)),dim=1).to(device)
    # pdb.set_trace()
    return augmented_image 



def Distance_Calculation_from_Source(model,img_stack,slice_selection_info):

    pre_mean,pre_var = slice_selection_info[0],slice_selection_info[1]
    
    model_2 = copy.deepcopy(model)
    model_2 = configure_model_init(model_2,1)

    dists = []

    for j in range(img_stack.shape[0]):

        img = img_stack[j]
        augmented_image = augmentation(img,20)
        # pdb.set_trace()
        out = model_2(augmented_image)
    
        #curr_stats have the current mean and var
        curr_mean, curr_var = get_feature_stat_auto(model_2)

        # any dists function can be calculated here: L1/L2/KL DIV
        dists.append(dist_function(curr_mean[-9:], curr_var[-9:], pre_mean[-9:], pre_var[-9:]).detach().cpu().tolist())

    dists = np.array(dists)
    dist_idx = np.argsort(dists)

    return  dists,dist_idx




def configure_model_init(model,momentum_):
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
                m.running_mean = torch.zeros([m.num_features]).cuda()
                m.running_var = torch.zeros([m.num_features]).cuda()
                m.momentum = momentum_  
    return model


def check_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            print(m.running_mean, m.running_var)


def get_feature_stat_auto(model):
    mean = []
    var = []
    for name,child in model.named_children():
        for name2, layer in child.named_modules():
            if isinstance(layer, nn.BatchNorm2d):
                mean.append(layer.running_mean.clone())
                var.append(layer.running_var.clone())      
    return mean, var

def dist_function(mean1, var1, mean2, var2, loss_func = 'KL'):

    final_sum = []
    # print(loss_func)  
    for m1, v1, m2, v2 in zip(mean1, var1, mean2, var2):
        sums = []
        eps = 1e-10
        for mm1, var1, mm2, var2 in zip(m1, v1, m2, v2):

            if loss_func=='KL':
                std1 = torch.sqrt(var1)
                std2 = torch.sqrt(var2)
                dis = torch.log(std2/(std1+eps)) + (std1**2 + (mm1 - mm2)**2)/((2*(std2**2))+eps) - 0.5
            elif loss_func=='l1':
                dis = torch.abs(mm1 - mm2)
            elif loss_func=='l2':
                dis = (mm1-mm2)**2
            sums.append(dis)
        final_sum.append(sum(sums)/len(sums))
    return sum(final_sum)/len(final_sum)




if __name__ == "__main__" :

    load_cfg_fom_args()




    data = torch.rand(60,3,256,256)
    pred = torch.rand(60,2,256,256)
    gt_label = torch.rand(60,2,256,256)
    selected_slices = [4,15,30,45]

    pdb.set_trace()

    # a = Interpixel_Active_loss(data,pred,gt_label,selected_slices,cfg)
    # b = Entropy_loss(pred)
    # c = Continuity_loss(pred)


    pdb.set_trace()
