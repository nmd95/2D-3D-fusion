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
from elbit_ds import Elbit_DS, torch_geometric_elbit_ds, Elbit_DS_multi
from models import PointNet_Graph_Naive_RGB_Parallel

if __name__ == '__main__':

    log_file_path = "/mnt/storage/datasets/elbit_pointcloud_segmentation/logging/results/naive_rgb_whole_ds_no_fps_2.txt"
    save_weights_path = "/mnt/storage/datasets/elbit_pointcloud_segmentation/logging/model_checkpoints/naive_rgb_whole_ds_no_fps_2"


    tg_ds_train = torch.load("/mnt/storage/datasets/elbit_pointcloud_segmentation/tg_datasets/raw/simple_no_fps_train_tg_ds.pkl")
    tg_ds_val = torch.load("/mnt/storage/datasets/elbit_pointcloud_segmentation/tg_datasets/raw/simple_no_fps_val_tg_ds.pkl")


    tg_dl_train = DataListLoader(tg_ds_train, shuffle=True, batch_size=32, num_workers=8, pin_memory=True)
    tg_dl_val = DataListLoader(tg_ds_val, shuffle=True, batch_size=64, num_workers=8, pin_memory=True)
    
    

    model = DataParallel(PointNet_Graph_Naive_RGB_Parallel(), device_ids=[0, 1, 2, 3, 4, 5, 6]) # return pointnet model that can run in parallel
    model = model.to('cuda:0')
    
    train_eval(model=model, train_dl=tg_dl_train, val_dl=tg_dl_val, epochs=1000, log_file_path=log_file_path, save_weights_path=save_weights_path)




