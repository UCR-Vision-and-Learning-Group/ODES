# -*- coding: utf-8 -*-
"""
Created on 09/07/2019

@author: Ali Emre Kavur
"""
import pydicom
import numpy as np
import glob, pdb
import cv2
# import SimpleITK as sitk
from scipy import ndimage
from sklearn.neighbors import KDTree
import torch
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion

def evaluate(Vref,Vseg,dicom_dir):
    dice=DICE(Vref,Vseg)
    ravd=RAVD(Vref,Vseg)
    [assd, mssd]=SSD(Vref,Vseg,dicom_dir)
    return dice, ravd, assd ,mssd



def surface_points(segmentation):
    """
    Extracts the surface points of a binary segmentation.
    Surface points are voxels that are True and have at least one False neighbor.

    Args:
        segmentation (np.ndarray): Binary segmentation mask.

    Returns:
        np.ndarray: Coordinates of surface voxels.
    """
    if not np.any(segmentation):  # If segmentation is empty
        return np.empty((0, segmentation.ndim), dtype=int)

    
    eroded = binary_erosion(segmentation)
    surface = segmentation ^ eroded  # XOR to find surface voxels
    return np.argwhere(surface)

def hausdorff_distance_95(V, V_ref):
    """
    Computes the 95th percentile Hausdorff Distance (HD95) between two binary segmentations.

    Args:
        V (np.ndarray): Predicted segmentation (binary mask).
        V_ref (np.ndarray): Ground truth segmentation (binary mask).

    Returns:
        float: 95th percentile Hausdorff Distance (HD95).
    """
    # Handle cases where both masks are empty

    V = np.array(V,dtype = bool)
    V_ref = np.array(V_ref,dtype = bool)
    
    if not np.any(V) and not np.any(V_ref):
        return 0.0

    # If only one is empty, return infinity (no overlap)
    if not np.any(V) or not np.any(V_ref):
        return float('inf')

    # Extract surface points
    surface_V = surface_points(V)
    surface_V_ref = surface_points(V_ref)

    # If either surface has no points, HD95 is infinite
    if surface_V.size == 0 or surface_V_ref.size == 0:
        return float('inf')

    # Compute pairwise distances between surface points
    distances_V_to_V_ref = cdist(surface_V, surface_V_ref)
    distances_V_ref_to_V = cdist(surface_V_ref, surface_V)

    # Minimum distances from each point to the other surface
    min_distances_V = np.min(distances_V_to_V_ref, axis=1)
    min_distances_V_ref = np.min(distances_V_ref_to_V, axis=1)

    # Combine distances
    all_min_distances = np.concatenate([min_distances_V, min_distances_V_ref])

    # Remove any potential inf or NaN values
    all_min_distances = all_min_distances[np.isfinite(all_min_distances)]

    # If no valid distances remain, return infinity
    if all_min_distances.size == 0:
        return float('inf')

    # Compute the 95th percentile
    hd95 = np.percentile(all_min_distances, 95)

    return hd95




def DICE(Vref,Vseg):
    dice=2*(torch.mul(Vref,Vseg)).sum()/(Vref.sum() + Vseg.sum()+1e-08)
    return dice

def RAVD(Vref,Vseg):
    ravd=(abs(Vref.sum() - Vseg.sum())/(Vref.sum())+1e-30)*100
    return ravd

def SSD(Vref,Vseg,dicom_dir):  
    struct = ndimage.generate_binary_structure(3, 1)  
    
    ref_border=Vref ^ ndimage.binary_erosion(Vref, structure=struct, border_value=1)
    ref_border_voxels=np.array(np.where(ref_border))
        
    seg_border=Vseg ^ ndimage.binary_erosion(Vseg, structure=struct, border_value=1)
    seg_border_voxels=np.array(np.where(seg_border))  
    
    ref_border_voxels_real=transformToRealCoordinates(ref_border_voxels,dicom_dir)
    seg_border_voxels_real=transformToRealCoordinates(seg_border_voxels,dicom_dir)    
  
    tree_ref = KDTree(np.array(ref_border_voxels_real))
    dist_seg_to_ref, ind = tree_ref.query(seg_border_voxels_real)
    tree_seg = KDTree(np.array(seg_border_voxels_real))
    dist_ref_to_seg, ind2 = tree_seg.query(ref_border_voxels_real)   
    
    assd=(dist_seg_to_ref.sum() + dist_ref_to_seg.sum())/(len(dist_seg_to_ref)+len(dist_ref_to_seg))
    mssd=np.concatenate((dist_seg_to_ref, dist_ref_to_seg)).max()    
    return assd, mssd

def transformToRealCoordinates(indexPoints,dicom_dir):
    """
    This function transforms index points to the real world coordinates
    according to DICOM Patient-Based Coordinate System
    The source: DICOM PS3.3 2019a - Information Object Definitions page 499.
    
    In CHAOS challenge the orientation of the slices is determined by order
    of image names NOT by position tags in DICOM files. If you need to use
    real orientation data mentioned in DICOM, you may consider to use
    TransformIndexToPhysicalPoint() function from SimpleITK library.
    """
    
    dicom_file_list=glob.glob(dicom_dir + '/*.dcm')
    dicom_file_list.sort()
    #Read position and orientation info from first image
    ds_first = pydicom.dcmread(dicom_file_list[0])
    img_pos_first=list( map(float, list(ds_first.ImagePositionPatient)))
    img_or=list( map(float, list(ds_first.ImageOrientationPatient)))
    pix_space=list( map(float, list(ds_first.PixelSpacing)))
    #Read position info from first image from last image
    ds_last = pydicom.dcmread(dicom_file_list[-1])
    img_pos_last=list( map(float, list(ds_last.ImagePositionPatient)))

    T1=img_pos_first
    TN=img_pos_last
    X=img_or[:3]
    Y=img_or[3:]
    deltaI=pix_space[0]
    deltaJ=pix_space[1]
    N=len(dicom_file_list)
    M=np.array([[X[0]*deltaI,Y[0]*deltaJ,(T1[0]-TN[0])/(1-N),T1[0]], [X[1]*deltaI,Y[1]*deltaJ,(T1[1]-TN[1])/(1-N),T1[1]], [X[2]*deltaI,Y[2]*deltaJ,(T1[2]-TN[2])/(1-N),T1[2]], [0,0,0,1]])

    realPoints=[]
    for i in range(len(indexPoints[0])):
        P=np.array([indexPoints[1,i],indexPoints[2,i],indexPoints[0,i],1])
        R=np.matmul(M,P)
        realPoints.append(R[0:3])

    return realPoints

def png_series_reader(dir):
    V = []
    png_file_list=glob.glob(dir + '/*.png')
    png_file_list.sort()
    for filename in png_file_list: 
        image = cv2.imread(filename,0)
        V.append(image)
    V = np.array(V,order='A')
    V = V.astype(bool)
    return V
