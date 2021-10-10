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
from torch_geometric.nn import DataParallel
from torch_geometric.data import DataListLoader


from model_utils import knn_dgl_to_pg, random_subsample
from train_val_test import train_epoch, test, train_eval
from elbit_ds import get_pg_dl
from models import PointNet, PointNet_VGG_mid_fusion, vgg_fuser
from elbit_ds import Elbit_DS, torch_geometric_elbit_ds, torch_geometric_elbit_ds_mid_fusion



if __name__ == '__main__':

# comment out if new data to create new pkl
  
  # elbit_ds = Elbit_DS("/home/students_1/finalproj/deep_graph_2d_3d_funsion/elbit_data/scenes/baqua_megido_flyby_2.7z", "/home/students_1/finalproj/deep_graph_2d_3d_funsion/elbit_data/images/baqua_megido_flyby_2.7z")
  # torch.save(elbit_ds, "/home/students_1/finalproj/deep_graph_2d_3d_funsion/tg_datasets/raw/huge.pkl")
  # val_len = int(len(elbit_ds)/3.0)
  # train_ds, val_ds = torch.utils.data.random_split(elbit_ds, [len(elbit_ds)-val_len, val_len])
  # train_batch_size = 1
  # val_batch_size = 1
  # train_dl = DataLoader(ds_to_torch_geometric(train_ds), batch_size=train_batch_size, shuffle=True)
  # val_dl = DataLoader(ds_to_torch_geometric(val_ds), batch_size=val_batch_size, shuffle=True)
  #train_dl_path = "/home/students_1/finalproj/deep_graph_2d_3d_funsion/elbit_data/pkles/baqua_megido_flyby_2.7z/train_dl_2s_filtered_ds.pkl"
  #val_dl_path = "/home/students_1/finalproj/deep_graph_2d_3d_funsion/elbit_data/pkles/baqua_megido_flyby_2.7z/train_dl_2s_filtered_ds.pkl"

  # torch.save(train_dl, train_dl_path)
  # torch.save(val_dl, val_dl_path)
  
  #  train_dl = get_pg_dl(train_dl_path, batch_size=6, shuffle=True) #return data_loader
  #  val_dl = get_pg_dl(val_dl_path, batch_size=6, shuffle=True) #return data_loader

  # train_eval(train_dl, val_dl, epochs=300)
  # finish to comment out
  
  # device = 'cuda:0'
  
  # #train_dl_path = "/home/students_1/finalproj/deep_graph_2d_3d_funsion/elbit_data/pkles/baqua_megido_flyby_2.7z/train_dl_2s_filtered_ds.pkl"
  # val_dl_path = "/home/students_1/finalproj/deep_graph_2d_3d_funsion/elbit_data/pkles/baqua_megido_flyby_2.7z/train_dl_2s_filtered_ds.pkl"

  # tg_ds = torch_geometric_elbit_ds(root="/home/students_1/finalproj/deep_graph_2d_3d_funsion/tg_datasets", torch_ds_pkl_file_name="huge.pkl", processed_data_dir="pdir_1")
  # torch.save(tg_ds, "/home/students_1/finalproj/deep_graph_2d_3d_funsion/tg_ds.pkl")
  # tg_ds = torch.load("/home/students_1/finalproj/deep_graph_2d_3d_funsion/tg_ds.pkl")
  # dll = DataListLoader(tg_ds, batch_size=3)
  # train_eval(dll, dll, epochs=5)

  # tg_ds_mid_fusion = torch_geometric_elbit_ds_mid_fusion(root="/home/students_1/finalproj/deep_graph_2d_3d_funsion/tg_datasets", torch_ds_pkl_file_name="huge.pkl", processed_data_dir="pdir_3")
  # torch.save(tg_ds_mid_fusion, "/home/students_1/finalproj/deep_graph_2d_3d_funsion/tg_ds_mid_fusion.pkl")
  tg_ds_mid_fusion = torch.load("/home/students_1/finalproj/deep_graph_2d_3d_funsion/tg_ds_mid_fusion.pkl")
  # dll = DataListLoader(tg_ds_mid_fusion, batch_size=1)
  dll = DataLoader(tg_ds_mid_fusion, batch_size=1)

  train_eval(dll, dll, epochs=5)





