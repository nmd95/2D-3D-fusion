# imports
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, join
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay
from imblearn.under_sampling import RandomUnderSampler
from keras.metrics import MeanIoU

from torch_geometric.data import Data, DataLoader
from torch_cluster import knn_graph
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import GraphNorm
from torch_cluster import fps
from collections import Counter
import torchgeometry as tgm
import dgl
import keras.metrics
from torch_geometric.utils import dropout_adj
from torch_geometric.utils import to_dense_batch

# function does coordinate normalizatio to image-size
def normlize_coords_to_grid(coords, H:int, W:int): # coords is of shape (#entries, 2), H, W are the hight and width of the image (repectively). 
  #normalize coords to [-1, 1] for grid-sampling
  if coords.shape[0] == 1:
    pix_x, pix_y = ((coords[:, 0]).round()),  ((coords[:, 1]).round())
  else:
    pix_x, pix_y = ((coords[:, 0]).squeeze().round()),  ((coords[:, 1]).squeeze().round())
  pix_x, pix_y = (2*(pix_x / W) - 1), (2*(pix_y / H) - 1)
  normalized_coords = torch.cat([pix_x.unsqueeze(1), pix_y.unsqueeze(1)], dim=1)
  return normalized_coords

''' function uses pytorch's "grid-sampler" to sample 2d-images ("input_2d") at pixel-locations specified by "projections".
 Batch vector is a tensor indicating the assignment of point-cloud instances to torch_geometric batches.'''
def batch_grid_sample_2d(input_2d, projections, batch_vector): # input_2d in of size (B, C, H, W), projections is of size (#batches*#points_per_batch, 2), i.e. pg's graph-batching-format
  proj_dense_batch, M = to_dense_batch(x=projections, batch=batch_vector) # shape should be (B, MAX#points, 2), M is mask of shape (B, MAX#points) to track zero-padding
  proj_dense_batch = proj_dense_batch.unsqueeze(1) # shape should now be (B, 1, MAX#points, 2)
  grid_sampled = F.grid_sample(input=input_2d, grid=proj_dense_batch, padding_mode='zeros') # shape of (B, C, 1, MAX#points)
  grid_sampled = grid_sampled.squeeze(2) # shape of (B, C, MAX#points)
  grid_sampled = grid_sampled.permute(0, 2, 1) # shape of (B, MAX#points, C)
  grid_sampled = grid_sampled.reshape(-1, grid_sampled.shape[2]) # shape of (B * MAX#points, C)
  M = M.reshape(-1, 1) # shape of (B * MAX#points, 1)
  grid_sampled = grid_sampled[M.squeeze(1),:] 

  return grid_sampled # (#batches*#points_per_batch, C)




'''This is a function for sampling 2d-images with pixel locations specified by "projections". This funciton down 
no use pytorch's grid-sampler.'''
def sample_2d_with_projections(input_2d, projections):
  input_2d = input_2d.squeeze()
  pix_x, pix_y = ((projections[:, 0] - 1).squeeze().round()),  ((projections[:, 1] - 1).squeeze().round())
  sampled_channels = input_2d[:, pix_y.long(), pix_x.long()].squeeze() 

  if len(sampled_channels.shape) < 2:
    sampled_channels = sampled_channels.unsqueeze(1)

  sampled_channels = sampled_channels.permute(1, 0)
  
  return sampled_channels
  
def sample_2d_batch(img_batch, projections_batch, batch_vector):

    def get_splits(orig, split_tensor):
        split_list = []
        prev_ind = 0
        curr_ind = 0

        for chunk in split_tensor:
            curr_ind += chunk.item()
            split_list.append(orig[prev_ind:curr_ind, :])
            prev_ind = curr_ind

        return split_list

    num_batches = img_batch.shape[0]
    counts = torch.bincount(batch_vector)
    split_list = get_splits(projections_batch, counts)

    samples_list = []
    for i in range(num_batches):
        temp_img = img_batch[i, :, :]
        samples_list.append(sample_2d_with_projections(temp_img, split_list[i]))
    return torch.cat(samples_list, dim=0)
    
# function to immitate the knn_graph function of torch_geometric using dgl (gpu efficient)
def knn_dgl_to_pg(pos, k, device:str):

  knn_g = dgl.knn_graph(pos, k=k)  # Each node has two predecessors
  e_tups = knn_g.edges(form='uv')
  edge_index = torch.stack((e_tups[1], e_tups[0])).to(device)

  return edge_index



def random_subsample(data, downsampling_factor=3.5):
  data = data
  index = (np.random.choice((data.pos.shape[0]-1), int(data.pos.shape[0]/downsampling_factor)))
  data.pos, data.batch, data.y = data.pos[index], data.batch[index], data.y[index]
  return data

  

