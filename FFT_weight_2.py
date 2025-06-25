# import random
# import copy, pdb
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms
# from skimage import morphology
# import tifffile


# from active.floating_region import FloatingRegionScore
# from active.spatial_purity import SpatialPurity
# from loss.negative_learning_loss import NegativeLearningLoss, Pseudo_label_Loss
# from utils.helper_functions import *
# from conf import cfg, load_cfg_fom_args
# from model.deeplab_v3 import seg_model


import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from copy import deepcopy
import pdb
import copy
from scipy.ndimage import zoom
from skimage.filters import threshold_otsu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def percentile_thresholding(heat_map, percentile=90):
    """
    Applies percentile-based thresholding to a 2D heat map.

    Parameters:
        heat_map (np.ndarray): 2D array with values between 0 and 1.
        percentile (float): Percentile value (0-100) to set the threshold.

    Returns:
        threshold (float): Computed percentile threshold.
        binary_map (np.ndarray): Thresholded binary map.
    """
    # Compute the percentile threshold
    threshold = np.percentile(heat_map, percentile)
    
    return threshold


def otsu_thresholding(heat_map):
    """
    Applies Otsu's thresholding to a 2D heat map.

    Parameters:
        heat_map (np.ndarray): 2D array with values between 0 and 1.

    Returns:
        threshold (float): Computed Otsu threshold.
        binary_map (np.ndarray): Thresholded binary map.
    """
    # Compute Otsu's threshold
    threshold = threshold_otsu(heat_map)
    
    return threshold




def compute_distances_whole(features: torch.Tensor) -> tuple:
    d, H, W = features.shape
    features_flat = features.permute(1, 2, 0).reshape(H * W, d)
    features_norm = F.normalize(features_flat, p=2, dim=1)
    cosine_similarity = torch.matmul(features_norm, features_norm.T)
    cosine_distance = 1 - cosine_similarity.reshape(H, W, H * W)

    y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    coords = torch.stack((y_coords, x_coords), dim=-1).reshape(H * W, 2).float()
    coords[:, 0] /= H
    coords[:, 1] /= W
    euclidean_distance = torch.norm(coords.unsqueeze(1) - coords.unsqueeze(0), dim=2).reshape(H, W, H * W)

    return cosine_distance, euclidean_distance


def normalized_gaussian(mean, cov, grid):
    # pdb.set_trace()
    rv = multivariate_normal(mean=mean, cov=cov)
    gaussian = rv.pdf(grid)
    return gaussian / np.max(gaussian)


def get_gaussian_weights(selected_pixels, cosine_distance, alpha,plot=0,file_name=None,draw=0):

    H, W = cosine_distance.shape[0],cosine_distance.shape[1]
    x, y = np.arange(W), np.arange(H)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    Z = np.zeros((H,W))

    for (row, col) in selected_pixels:
        local_distance = cosine_distance[row, col].view(H, W)
        row_min, row_max = np.clip([row - 2, row + 3], 0, H)
        col_min, col_max = np.clip([col - 2, col + 3], 0, W)
        local_patch = local_distance[row_min:row_max, col_min:col_max]
        cos_dist = torch.mean(local_patch).item()

        # if draw==1:
        #     sigma = alpha
        # else:
        #     sigma = max(1e-3, alpha * cos_dist)
        sigma = max(1e-3, alpha * cos_dist)
        # sigma = max(1e-3, 5)
        cov = [[sigma ** 2, 0], [0, sigma ** 2]]

        g_map = normalized_gaussian([col, row], cov, pos)
        # z1 = 0.5*(1 - g_map)
        # z1[row, col] = 1  
        
        Z += g_map

    if plot==1:

        # Plotting

        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
        plt.colorbar(contour, label='Intensity')

        # Mark peak locations

        peak_locations = np.array(selected_pixels)
        plt.scatter(peak_locations[:, 1], peak_locations[:, 0], color='red', marker='x', s=100, label='Selected Pixels')
        plt.title('Combined Gaussian Curves with Peak Value 1 at Selected Pixels')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        # plt.show()

    return Z

def highlight_top_n(image, k_percent, save_path="highlighted_image.png"):
    # Flatten the image to find top N values and their indices
    image = image/np.max(image)
    n = max(1, int(k_percent*image.shape[0]*image.shape[1]/ 100.0 ))
    flat_indices = np.argpartition(image.flatten(), -n)[-n:]
    row_indices, col_indices = np.unravel_index(flat_indices, image.shape)

    # Create a single figure with the heatmap and red crosses
    plt.figure(figsize=(8, 6))
    plt.title(f"Heatmap with Top {n} Values Highlighted")
    plt.imshow(image, cmap='viridis')
    plt.scatter(col_indices, row_indices, color='red', marker='x', s=100, label='Top Values')
    plt.colorbar()
    plt.legend()

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def highlight_top_n_two_images(image1, image2, k_percent, save_path="highlighted_images.png"):
    def get_top_indices(image, k_percent):
        image_normalized = image / np.max(image)
        n = max(1, int(k_percent * image.shape[0] * image.shape[1] / 100.0))
        flat_indices = np.argpartition(image_normalized.flatten(), -n)[-n:]
        row_indices, col_indices = np.unravel_index(flat_indices, image.shape)
        return image_normalized, row_indices, col_indices, n

    # Process both images
    image1_norm, row_indices1, col_indices1, n1 = get_top_indices(image1, k_percent)
    image2_norm, row_indices2, col_indices2, n2 = get_top_indices(image2, k_percent)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))

    # Plot for the first image
    axes[0].imshow(image1_norm)
    axes[0].scatter(col_indices1, row_indices1, color='red', marker='x', s=100, label='Top Values')
    axes[0].set_title(f"Original Uncertainity Map")
    axes[0].legend()

    # Plot for the second image
    axes[1].imshow(image1_norm)
    axes[1].scatter(col_indices2, row_indices2, color='red', marker='x', s=100, label='Top Values')
    axes[1].set_title(f"After Weighting Uncertainity Map")
    axes[1].legend()

    # Add colorbar that applies to both subplots
    # cbar = fig.colorbar(axes[0].images[0], ax=axes, orientation='vertical', fraction=0.025, pad=0.04)
    # cbar.set_label("Normalized Intensity")

    # Save and show the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example usage:
# image1 = np.random.rand(100, 100)
# image2 = np.random.rand(100, 100)
# highlight_top_n_two_images(image1, image2, k_percent=5)




def get_weight_map(heat_map, threshold, features, k_percent, alpha_cos=10, alpha_eu=30, plot=0):
    """
    Generates Gaussian weight maps based on cosine and Euclidean distances for selected pixels where the heat_map values exceed a threshold.

    Args:
        heat_map (torch.Tensor): Heat map with shape (H, W).
        threshold (float): Threshold to select relevant pixels.
        features (torch.Tensor): Feature tensor with shape (d, H, W).
        k_percent (float): Percentage of points to select in farthest-first traversal.
        alpha_cos (float): Scaling factor for cosine distance-based Gaussian weights.
        alpha_eu (float): Scaling factor for Euclidean distance-based Gaussian weights.
        plot (int): Flag to enable plotting and saving weight maps.

    Returns:
        tuple: Gaussian weight maps for cosine and Euclidean distances.
    """

    
    # pdb.set_trace()
    cosine_distance_whole, euclidean_distance_whole = compute_distances_whole(features)
    # print(cosine_distance_whole.shape)

    heat_map = torch.from_numpy(np.abs(heat_map))
    heat_map = heat_map.unsqueeze(0).unsqueeze(0)

    # Use average pooling with kernel_size and stride of 4 to downsample to (64, 64)
    heat_map = F.avg_pool2d(heat_map, kernel_size=4, stride=4)

    # Convert back to numpy array and remove batch and channel dimensions
    heat_map = heat_map.squeeze()
    heat_map = heat_map/heat_map.max()
    heat_map_numpy = copy.deepcopy(heat_map).numpy()
    threshold = percentile_thresholding(heat_map_numpy, percentile=100 - 3*k_percent)

    if threshold<0:
        threshold=0


    H, W = heat_map.shape

    # Get mask of valid points exceeding the threshold
    valid_mask = heat_map > threshold

    p = valid_mask.sum()/(H*W)
    # print(threshold,p)
    if not valid_mask.any():
        # print(p)
        print("No Weight Involved")
        # pdb.set_trace()
        return np.ones_like(heat_map_numpy),cosine_distance_whole, euclidean_distance_whole
        # valid_mask = heat_map > 0.5
        # raise ValueError("No points exceed the given threshold.")

    # Filter features and corresponding coordinates based on valid mask
    y_coords, x_coords = torch.where(valid_mask)
    selected_features = features[:, y_coords, x_coords]  # Shape: (d, N)
    first_idx = torch.argmax(heat_map[y_coords,x_coords]).item()

    # Compute distances for filtered points
    features_norm = F.normalize(selected_features.T, p=2, dim=1)  # Shape: (N, d)
    cosine_similarity = torch.matmul(features_norm, features_norm.T)
    cosine_distance = 1 - cosine_similarity
    cosine_distance = cosine_distance/cosine_distance.max()
    

    coords = torch.stack((y_coords.float() / H, x_coords.float() / W), dim=1)
    euclidean_distance = torch.norm(coords.unsqueeze(1) - coords.unsqueeze(0), dim=2)
    euclidean_distance = euclidean_distance/euclidean_distance.max()
    # print('euclidean',euclidean_distance.shape)

    # pdb.set_trace()

    def farthest_first_traversal(distance_matrix, k_percent, first_idx=None):
        N = distance_matrix.shape[0]
        k = max(1, int(k_percent*H*W/ 100.0 ))
        # print(N,k)
        
        # pdb.set_trace()
        if first_idx is None:
            first_idx = torch.randint(0, N, (1,), device=distance_matrix.device).item()

        selected_indices = [first_idx]
        min_distances = distance_matrix[first_idx].clone()

        for _ in range(1, k):
            farthest_index = torch.argmax(min_distances).item()
            selected_indices.append(farthest_index)
            min_distances = torch.minimum(min_distances, distance_matrix[farthest_index])

        return [(y_coords[idx].item(), x_coords[idx].item()) for idx in selected_indices]

    # Cosine-based weights
    selected_pixels_cosine = farthest_first_traversal(cosine_distance, k_percent, first_idx)
    # gaussian_weights_cosine = get_gaussian_weights(selected_pixels_cosine, cosine_distance_whole, alpha_cos, plot, 'heatmap/cosine_selected_pts.png')
    gaussian_weights_cosine = get_gaussian_weights(selected_pixels_cosine, cosine_distance_whole, 1, plot, 'heatmap/cosine_selected_pts.png',draw=1)

    # Euclidean-based weights
    selected_pixels_euclidean = farthest_first_traversal(euclidean_distance, k_percent, first_idx)
    # pdb.set_trace()
    # gaussian_weights_euclidean = get_gaussian_weights(selected_pixels_euclidean, euclidean_distance_whole, alpha_eu, plot, 'heatmap/euclidean_selected_pts.png')
    gaussian_weights_euclidean = get_gaussian_weights(selected_pixels_euclidean, euclidean_distance_whole, 1, plot, 'heatmap/euclidean_selected_pts.png',draw=1)

    weight = (gaussian_weights_cosine*gaussian_weights_euclidean)
    weighted_heat_map = heat_map_numpy*weight
    
    # print(len(selected_pixels_cosine))
    if plot:

        def plot_weight_map(Z, file_name):
            plt.figure(figsize=(10, 8))
            X, Y = np.meshgrid(np.arange(W), np.arange(H))
            contour = plt.contourf(X, Y, Z/np.max(Z), levels=50, cmap='viridis')
            # contour = plt.contourf(X, Y, Z/np.max(Z), levels=50, cmap='Greys')
            plt.colorbar(contour, label='Intensity')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            plt.close()

        plot_weight_map(gaussian_weights_cosine, 'heatmap/map_cosine_dist.png')
        plot_weight_map(gaussian_weights_euclidean, 'heatmap/map_euclidean_dist.png')
        plot_weight_map(weight, 'heatmap/combined_weight.png')
        # pdb.set_trace()

        plot_weight_map(copy.deepcopy(features[0]).cpu().numpy(), 'heatmap/feature_heatmap.png')
        plot_weight_map(heat_map_numpy, 'heatmap/initial_heatmap.png')
        # plot_weight_map(valid_mask.numpy(),'heatmap/threshold.png')
        # weighted_heat_map = weighted_heat_map/np.max(weighted_heat_map)
        plot_weight_map(weighted_heat_map, 'heatmap/weighted_heatmap.png')
        plot_weight_map(valid_mask.numpy(),'heatmap/threshold.png')



        highlight_top_n(weight,k_percent,'heatmap/weight.png')
        highlight_top_n_two_images(heat_map_numpy, weighted_heat_map, k_percent, save_path="heatmap/highlighted_images.png")

        # highlight_top_n(heat_map_numpy,k_percent,'heatmap/initial_heatmap.png')
        # highlight_top_n(heat_map_numpy,k_percent,'heatmap/weighted_heatmap.png')

    # return gaussian_weights_cosine, gaussian_weights_euclidean, cosine_distance_whole, euclidean_distance_whole
    # pdb.set_trace()
    return weight, cosine_distance_whole, euclidean_distance_whole

def farthest_first_with_reference_distance(data: torch.Tensor, distance_from_reference: torch.Tensor, k_percent: float, device='cpu'):
    """
    Select k% samples from data using farthest first traversal considering:
    1. Precomputed distance from a reference (provided as an array).
    2. Diversity among selected samples.
    3. Avoiding repeated selections.

    Args:
        data (torch.Tensor): Input tensor of shape (N, d, H, W).
        distance_from_reference (torch.Tensor): Precomputed distances of shape (N,).
        k_percent (float): Percentage of samples to select (between 0 and 100).
        device (str): Device to perform computations ('cpu' or 'cuda').

    Returns:
        selected_indices (list): Indices of selected samples.
    """
    N = data.shape[0]

    # pdb.set_trace()
    k = max(1, int((k_percent / 100) * N))  # Number of samples to select

    # Flatten and normalize data: (N, d*H*W)
    data_flat = data.view(N, -1).to(device)
    data_flat = torch.nn.functional.normalize(data_flat, dim=1)

    # Move distance array to device
    distances = distance_from_reference.to(device).clone()

    # Initialize selected indices with the farthest point from the reference
    selected_indices = []
    available_indices = set(range(N))

    # Select the first sample: farthest from the reference
    farthest_idx = torch.argmax(distances).item()
    selected_indices.append(farthest_idx)
    available_indices.remove(farthest_idx)

    for _ in range(k - 1):
        # Compute cosine distance to the last selected sample
        last_selected = data_flat[selected_indices[-1]].unsqueeze(0)  # (1, d*H*W)
        cosine_sim_selected = torch.matmul(data_flat, last_selected.T).squeeze()  # (N,)
        cosine_dist_selected = 1 - cosine_sim_selected  # Cosine distance
        total_distance = cosine_dist_selected + distance_from_reference

        # Update distances: minimum distance between reference array and selected samples
        distances = torch.minimum(distances, total_distance)

        # Select the farthest available index
        sorted_indices = torch.argsort(distances, descending=True).tolist()
        farthest_idx = next(idx for idx in sorted_indices if idx in available_indices)

        # Add to selected and remove from available
        selected_indices.append(farthest_idx)
        available_indices.remove(farthest_idx)

    # print(selected_indices)
    return selected_indices

def expand_labels_with_cosine_similarity(A, cosine_similarity, threshold=0.9):
    """
    Expand labels in A by propagating labels from labeled pixels to unlabeled ones based on cosine similarity.

    Args:
        A (torch.Tensor): Tensor of shape (H, W) with labels in [0, C] and 255 for unlabeled pixels.
        cosine_similarity (torch.Tensor): Tensor of shape (H, W, H*W) representing cosine similarity.
        threshold (float): Similarity threshold to propagate labels.

    Returns:
        torch.Tensor: Enhanced annotated image of shape (H, W).
    """
    # A = torch.from_numpy(A)
    
    # pdb.set_trace()
    A = F.interpolate(A.unsqueeze(0).unsqueeze(0), size=(64, 64), mode='nearest-exact').squeeze()
    
    # A = zoom(A.cpu(), (1/4, 1/4), order=0) 
    # A = torch.from_numpy(A).to(device)
    # A = F.avg_pool2d(A.unsqueeze(0).unsqueeze(0), kernel_size=4, stride=4).round()
    # A = A.squeeze()
    # pdb.set_trace()

    H, W = A.shape
    A_expanded = A.clone()
    
    # Flatten the label matrix for easier assignment
    A_flat = A_expanded.view(-1)

    # Get indices of labeled pixels
    labeled_indices = torch.nonzero(A_expanded != 255, as_tuple=False)

    for idx in labeled_indices:
        h, w = idx.tolist()
        label = A_expanded[h, w].item()

        # Retrieve the similarity vector for the current labeled pixel
        pixel_similarity = cosine_similarity[h, w]  # Shape: (H*W,)

        # Find indices of pixels with similarity above threshold
        high_sim_indices = torch.nonzero(pixel_similarity > threshold, as_tuple=False).squeeze()

        # pdb.set_trace()

        if high_sim_indices.numel() > 0:
            # Filter indices that correspond to unlabeled pixels
            unlabeled_indices = high_sim_indices[A_flat[high_sim_indices] == 255]

            # Assign the label to these unlabeled pixels
            A_flat[unlabeled_indices] = label

    # pdb.set_trace()
    A_expanded = F.interpolate(A_expanded.unsqueeze(0).unsqueeze(0), size=(256,256), mode='nearest-exact').squeeze()


    return A_expanded




if __name__ == "__main__":


    # load_cfg_fom_args()
    # pdb.set_trace()
    # model = seg_model(num_class=5)
    # data = torch.rand(10, 3, 256, 256)
    # label = torch.randint(0,2,(10, 2, 256, 256))
    
    # output,feature,combined_feature = model(data,require_feature=True)

    # pdb.set_trace()

    # cosine_dist, euclidean_dist = compute_distances_(combined_feature[5])
    # selected_pixels = farthest_first_traversal(cosine_dist, k_percent=5)
    # gaussian_weights = get_gaussian_weights(selected_pixels, cosine_dist,plot=1)

    # pdb.set_trace()
    # get_weight_map(combined_feature[5],1)

    # slice_info = [torch.rand(10), torch.rand(10)]

    # tgt_mask_pixel = pixel_selection_equal(None, data, label, slice_info, cfg)
    # tgt_mask_region = region_selection_equal(None, data, label, slice_info, cfg)
    # plot_heat_map(tgt_mask_pixel.cpu().numpy(), 'selected_pixels.tiff')
    # plot_heat_map(tgt_mask_region.cpu().numpy(), 'selected_regions.tiff')

    # features = torch.rand(8, 64, 64)
    # cosine_dist, euclidean_dist = compute_distances(features)
    # selected_pixels = farthest_first_traversal(cosine_dist, k_percent=5)
    # gaussian_weights = get_gaussian_weights(selected_pixels, cosine_dist)

    # Sample data: 50 samples, each with shape (3, 32, 32)
    N, d, H, W = 50, 3, 32, 32
    data = torch.randn(N, d, H, W)

    # Precomputed distance array (N,)
    distance_from_reference = torch.rand(N)  # Random distances for demonstration

    # Select 20% samples
    k_percent = 20
    selected_indices = farthest_first_with_reference_distance(data, distance_from_reference, k_percent)
    print("Selected Indices:", selected_indices)

    # Example Usage
    H, W, d = 64, 64, 64  # Dimensions
    heat_map = torch.rand(H, W)  # Random heat map
    features = torch.rand(d, H, W)  # Random feature map

    # Run the function with example parameters
    gaussian_weights_cosine, gaussian_weights_euclidean = get_weight_map(
        heat_map=heat_map,
        threshold=0.9,  # Threshold for valid pixels
        features=features,
        k_percent=1,  # 5% of valid pixels selected
        alpha_cos=10,
        alpha_eu=40,
        plot=1
    )


    # Create a sample 256x256 tensor
    x = torch.rand(256, 256)

    # Reshape to (1, 1, 256, 256) - 1 batch, 1 channel
    x_unsq = x.unsqueeze(0).unsqueeze(0)

    # Downsample to 64x64 using bilinear interpolation
    x_interp = F.interpolate(x_unsq, size=(64, 64), mode='bilinear', align_corners=False)

    # Remove the batch and channel dimensions
    x_interp = x_interp.squeeze(0).squeeze(0)