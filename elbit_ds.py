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
import os.path as osp
from torch_geometric.data import Dataset as TG_DS
from model_utils import normlize_coords_to_grid
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer, seed_everything)
from torch_geometric.data import DataListLoader
from typing import Optional
from PIL import Image


class elbit_ds_lit(LightningDataModule):
    def __init__(self, data_dir, ds_name:str, batch_size:int):
        super().__init__()
        self.data_dir = data_dir
        self.train_ds_path = data_dir + '/' + ds_name + '_train_tg_ds.pkl'
        self.val_ds_path = data_dir + '/' + ds_name + '_val_tg_ds.pkl'
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):

        self.tg_ds_test = torch.load(self.val_ds_path)
        self.tg_ds_val = torch.load(self.val_ds_path)
        self.tg_ds_train = torch.load(self.train_ds_path)

    def train_dataloader(self):
        return DataLoader(self.tg_ds_train, shuffle=True, batch_size=self.batch_size, num_workers=40, prefetch_factor=32, pin_memory=True)


    def val_dataloader(self):
        return DataLoader(self.tg_ds_val, shuffle=True, batch_size=self.batch_size, num_workers=40, prefetch_factor=32, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.tg_ds_val, shuffle=True, batch_size=self.batch_size, num_workers=40, prefetch_factor=32, pin_memory=True)


# parses the text files which contain the points-stats
def parse_data(path:str):
  file = open(path, mode = 'r', encoding = 'utf-8-sig')
  lines = file.readlines()
  file.close()

  my_list = []
  for line in lines:
    line = line.split(',')
    line = [i.strip('\n') for i in line]
    line = [float(i) for i in line]
    my_list.append(line)

  return my_list

# returns two tensors: one representing the pointcoulds of the scene, and one for the corresponding image
# each row in the pointclouds tensors represents a differen point, and the columns have the following structure: (x, y, z, class, pixLocs(x), pixLocs(y), time)

def get_scene_tensor(txt_scene_path:str, img_dir_path:str):

  im_to_tensor = T.ToTensor()

  data_list = parse_data(txt_scene_path)

  img_num = str(int(data_list[0][-1]))
  img_path = img_dir_path + "/" + img_num + ".jpeg"
   #r"C:\Users\System-Pc\Desktop\ybear.jpg"
  img_tensor = im_to_tensor(Image.open(img_path))

  data_list = [line[:-1] for line in data_list]

  points_tensor = torch.FloatTensor(data_list)

  return points_tensor, img_tensor

# returns two tensors: one representing the pointcoulds of the scene, and one for the corresponding image
# each row in the pointclouds tensors represents a differen point, and the columns have the following structure: (x, y, z, class, pixLocs(x), pixLocs(y), time, R, G, B)

def get_scene_tensor_2(txt_scene_path:str, img_dir_path:str):

  im_to_tensor = T.ToTensor()

  data_list = parse_data(txt_scene_path)

  img_num = str(int(data_list[0][-1]))
  img_path = img_dir_path + "/" + img_num + ".jpeg"
  #r"C:\Users\System-Pc\Desktop\ybear.jpg"
  img_tensor = im_to_tensor(Image.open(img_path))

  data_list = [line[:-1] for line in data_list]
  points_tensor = torch.FloatTensor(data_list)

  # adding RGB to each 3 point (sampling from 2D image)
  pix_x, pix_y = ((points_tensor[:, 4] - 1).squeeze().round()),  ((points_tensor[:, 5] - 1).squeeze().round())
  
  RGB = img_tensor[:, pix_y.long(), pix_x.long()].squeeze() #  RGB = img_tensor[:, pix_x.long(), pix_y.long()].squeeze()
  if len(RGB.shape) < 2:
    RGB = RGB.unsqueeze(1)
  
  RGB = RGB.permute(1, 0)
  points_tensor = torch.cat([points_tensor, RGB], dim=1)
  index = torch.from_numpy(np.random.choice((points_tensor.shape[0]-1), int(points_tensor.shape[0]/30.0))) # temporary - to be changed !
  # index = fps(points_tensor[:, 0:3], ratio=0.5)
  return points_tensor[index, :], img_tensor

def get_scene_tensor_3(txt_scene_path:str, img_dir_path:str):

  im_to_tensor = T.ToTensor()

  data_list = parse_data(txt_scene_path)

  img_num = str(int(data_list[0][-1]))
  img_path = img_dir_path + "/" + img_num + ".jpeg"
  #r"C:\Users\System-Pc\Desktop\ybear.jpg"
  img_tensor = im_to_tensor(Image.open(img_path))

  data_list = [line[:-1] for line in data_list]
  points_tensor = torch.FloatTensor(data_list)

  # adding RGB to each 3 point (sampling from 2D image)
  pix_x, pix_y = ((points_tensor[:, 4] - 1).squeeze().round()),  ((points_tensor[:, 5] - 1).squeeze().round())
  
  RGB = img_tensor[:, pix_y.long(), pix_x.long()].squeeze() #  RGB = img_tensor[:, pix_x.long(), pix_y.long()].squeeze()
  if len(RGB.shape) < 2:
    RGB = RGB.unsqueeze(1)
  
  points_tensor = torch.cat([points_tensor, RGB.permute(1, 0)], dim=1)
  # index = torch.from_numpy(np.random.choice((points_tensor.shape[0]-1), int(points_tensor.shape[0]/30.0))) # temporary - to be changed !
  # index = fps(points_tensor[:, 0:3], ratio=0.5)

  return points_tensor, img_tensor




class torch_geometric_elbit_ds(TG_DS): 
  def __init__(self, root, torch_ds_pkl_file_name ,processed_data_dir, val:bool, transform=None, pre_transform=None):
    

    self.torch_ds_pkl_file_name = torch_ds_pkl_file_name
    self.processed_data_dir = processed_data_dir
    self.val = val

    super(torch_geometric_elbit_ds, self).__init__(root, transform, pre_transform)

  @property
  def raw_file_names(self):
        return [self.torch_ds_pkl_file_name, self.processed_data_dir]

  @property
  def processed_file_names(self):
    self.data = torch.load(self.raw_paths[0])
    return [f'data_{i}.pt' for i in range(len(self.data))]

  def download(self):
    pass

  def process(self):
    self.data = torch.load(self.raw_paths[0])

    for i, sample in tqdm(enumerate(self.data)):

      # in the future add rgb features via sample[1] (the image) to the x argument in the Data object

      # if self.val == False:
      #   # index = torch.from_numpy(np.random.choice((sample[0].shape[0]-1), int(sample[0].shape[0]/self.downsample_factor))) # temporary - to be changed !
      #   index = fps(sample[0][:, 0:3], ratio=0.03)
      #   sample = (sample[0][index, :], sample[1])
      #   data = Data(pos=sample[0][:, :3], y=sample[0][:, 3:4], x=sample[0][:, 6:10])
      # else:
      #   index = fps(sample[0][:, 0:3], ratio=0.03)
      #   sample = (sample[0][index, :], sample[1])
      #   data = Data(pos=sample[0][:, :3], y=sample[0][:, 3:4], x=sample[0][:, 6:10])
      # torch.save(data, osp.join(self.raw_paths[1], 'data_{}.pt'.format(i)))

      data = Data(pos=sample[0][:, :3], y=sample[0][:, 3:4], x=sample[0][:, 6:10])
      torch.save(data, osp.join(self.raw_paths[1], 'data_{}.pt'.format(i)))

  def len(self):
    return len(self.data)

  def get(self, idx):
    data = torch.load(osp.join(self.raw_paths[1], 'data_{}.pt'.format(idx)))
    return data



class torch_geometric_elbit_ds_mid_fusion(TG_DS): 
  def __init__(self, root, torch_ds_pkl_file_name ,processed_data_dir, val:bool, transform=None, pre_transform=None, sample_factor=4.0):
    

    self.torch_ds_pkl_file_name = torch_ds_pkl_file_name
    self.processed_data_dir = processed_data_dir
    self.val = val
    self.sample_factor = sample_factor

    super(torch_geometric_elbit_ds_mid_fusion, self).__init__(root, transform, pre_transform)

  @property
  def raw_file_names(self):
        return [self.torch_ds_pkl_file_name, self.processed_data_dir]

  @property
  def processed_file_names(self):
    self.data = torch.load(self.raw_paths[0])
    return [f'data_{i}.pt' for i in range(len(self.data))]

  def download(self):
    pass

  def process(self):
    self.data = torch.load(self.raw_paths[0])

    for i, sample in tqdm(enumerate(self.data)):
      

      # if self.val == False:
      #   # index = torch.from_numpy(np.random.choice((sample[0].shape[0]-1), int(sample[0].shape[0]/self.downsample_factor))) # temporary - to be changed !
      #   index = fps(sample[0][:, 0:3], ratio=0.3)
      #   sample = (sample[0][index, :], sample[1])
      #   data = Data(pos=sample[0][:, :3], y=sample[0][:, 3:4], img_2d=sample[1].unsqueeze(0), projections=sample[0][:, 4:6])
      
      # else:
      #   data = Data(pos=sample[0][:, :3], y=sample[0][:, 3:4], img_2d=sample[1].unsqueeze(0), projections=sample[0][:, 4:6])
      
      H, W = sample[1].shape[1], sample[1].shape[2] #correction- need to change shape[0] to shape[1] and shape[1] to shape[2]
      projections = normlize_coords_to_grid(sample[0][:, 4:6], H, W)
      img_pil = T.ToPILImage()(sample[1])
      img_pil = img_pil.resize((round(img_pil.size[0] / self.sample_factor), round(img_pil.size[1] / self.sample_factor)), Image.NEAREST)
      img_tensor = T.ToTensor()(img_pil)
      data = Data(pos=sample[0][:, :3], y=sample[0][:, 3:4], img_2d=img_tensor.unsqueeze(0), projections=projections)
      
      # data = Data(pos=sample[0][:, :3], y=sample[0][:, 3:4], img_2d=sample[1].unsqueeze(0), projections=sample[0][:, 4:6])
      torch.save(data, osp.join(self.raw_paths[1], 'data_{}.pt'.format(i)))

  def len(self):
    return len(self.data)

  def get(self, idx):
    data = torch.load(osp.join(self.raw_paths[1], 'data_{}.pt'.format(idx)))
    return data

        
        
class Elbit_DS(Dataset):

    def __init__(self, scenes_dir_path, img_dir_path):
     """
     Args:
            scenes_dir_path (string): Path the dir with all scenes as txt files
            img_dir_path (string): path of directory with all the images.
        """

     self.scenes_dir_path = scenes_dir_path
     self.img_dir_path = img_dir_path
     self.scene_names = [f for f in listdir(self.scenes_dir_path) if isfile(join(self.scenes_dir_path, f))]

    def __len__(self):
        return len(self.scene_names)

    def __getitem__(self, idx):

      scene_name = self.scene_names[idx]
      scene_path = self.scenes_dir_path + "/" + scene_name

      return get_scene_tensor_3(scene_path, self.img_dir_path)


# usage:  multi_ds = Elbit_DS_multi([ds_1=Elbit_DS(...), ds_2=Elbit_DS(...), ds_3=Elbit_DS(...)])
class Elbit_DS_multi(Dataset):

    def __init__(self, elbit_sets:list):
        self.elbit_sets = elbit_sets
        self.sets_lens = [len(ds) for ds in self.elbit_sets]
        self.top_indices = [length-1 for length in self.sets_lens]

    def __len__(self):
        return sum(self.sets_lens)

    def __getitem__(self, idx):
      offset = 0
      offset_prev = 0
      for k, top_idx in enumerate(self.top_indices):

        offset += (top_idx + 1)
        if idx < offset:
          idx = idx - offset_prev
          return self.elbit_sets[k][idx]
        else:
          offset_prev += (top_idx + 1)

      raise IndexError("idx not in range of any ds")
# def ds_to_torch_geometric(ds, downsampling_ratio=0.25): # downsampling is done via Fartherst Point Sampling (FPS)
#   data_list = []
#   for sample in tqdm(ds):
#     # in the future add rgb features via sample[1] (the image) to the x argument in the Data object
#     data = Data(pos=sample[0][:, :3], y=sample[0][:, 3:4])
#     # index = fps(data.pos, ratio=downsampling_ratio)
#     index = np.random.choice((data.pos.shape[0]-1), int(data.pos.shape[0]/30.0)) # temporary - to be changed !
#     data.pos = data.pos[index]
#     data.y = data.y[index]
#     data_list.append(data)
#   return data_list

def ds_to_torch_geometric(ds, downsampling_ratio=0.25): # downsampling is done via Fartherst Point Sampling (FPS)
  data_list = []
  for sample in tqdm(ds):
    # in the future add rgb features via sample[1] (the image) to the x argument in the Data object
    data = Data(pos=sample[0][:, :3], y=sample[0][:, 3:4], x=sample[0][:, 7:10])
    # index = fps(data.pos, ratio=downsampling_ratio)
    index = np.random.choice((data.pos.shape[0]-1), int(data.pos.shape[0]/10.0)) # temporary - to be changed !
    data.pos = data.pos[index]
    data.y = data.y[index]
    data.x = data.x[index]
    data_list.append(data)
  return data_list



def get_pg_dl(path:str, batch_size:int, shuffle:bool):

  dl = torch.load(path)
  dl = DataLoader(dl.dataset, batch_size=batch_size, shuffle=shuffle)

  return dl




