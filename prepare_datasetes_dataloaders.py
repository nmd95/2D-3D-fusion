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
from elbit_ds import Elbit_DS, torch_geometric_elbit_ds, Elbit_DS_multi, torch_geometric_elbit_ds_mid_fusion
from models import PointNet_Graph_Naive_RGB_Parallel

import glob
import os

if __name__ == '__main__':

    raw_base_path = "/mnt/storage/datasets/elbit_pointcloud_segmentation/tg_datasets/raw/"
    ds_name = "fusion_with_imgs_sample_factor_4"
    tg_datasets_path = "/mnt/storage/datasets/elbit_pointcloud_segmentation/tg_datasets"

    os.mkdir("/mnt/storage/datasets/elbit_pointcloud_segmentation/tg_datasets/raw/" +ds_name+ "_train")
    os.mkdir("/mnt/storage/datasets/elbit_pointcloud_segmentation/tg_datasets/raw/" +ds_name+ "_val")

    all_data_folders_1 = glob.glob("/mnt/storage/datasets/elbit_pointcloud_segmentation/new_play_ground/haim_nimrod_play_ground_1/*")
    all_data_folders_2 = glob.glob("/mnt/storage/datasets/elbit_pointcloud_segmentation/new_play_ground/haim_nimrod_play_ground_2/*")
    all_data_folders_3 = glob.glob("/mnt/storage/datasets/elbit_pointcloud_segmentation/new_play_ground/haim_nimrod_play_ground_3/*")
    all_data_folders_4 = glob.glob("/mnt/storage/datasets/elbit_pointcloud_segmentation/new_play_ground/haim_nimrod_play_ground_4/*")
    all_data_folders_5 = glob.glob("/mnt/storage/datasets/elbit_pointcloud_segmentation/new_play_ground/haim_nimrod_play_ground_5/*")

    all_data_folders = all_data_folders_1 + all_data_folders_2 + all_data_folders_3 + all_data_folders_4 + all_data_folders_5

    ds_list = [Elbit_DS(path + '/scenes', path + '/images') for path in all_data_folders]

    multi_ds = Elbit_DS_multi(ds_list)

    # _, multi_ds = torch.utils.data.random_split(multi_ds, [len(multi_ds)-1000, 1000])

    val_len = int(len(multi_ds)/3)
    train_ds, val_ds = torch.utils.data.random_split(multi_ds, [len(multi_ds)-val_len, val_len])
    torch.save(train_ds, raw_base_path + ds_name + "_train.pkl")
    torch.save(val_ds, raw_base_path + ds_name + "_val.pkl")   
    # tg_ds_train = torch_geometric_elbit_ds(val=False, root=tg_datasets_path, torch_ds_pkl_file_name=ds_name + "_train.pkl", processed_data_dir=ds_name + "_train")
    # tg_ds_val = torch_geometric_elbit_ds(val=True, root=tg_datasets_path, torch_ds_pkl_file_name=ds_name + "_val.pkl", processed_data_dir=ds_name + "_val")
    tg_ds_train = torch_geometric_elbit_ds_mid_fusion(val=False, root=tg_datasets_path, torch_ds_pkl_file_name=ds_name + "_train.pkl", processed_data_dir=ds_name + "_train")
    tg_ds_val = torch_geometric_elbit_ds_mid_fusion(val=True, root=tg_datasets_path, torch_ds_pkl_file_name=ds_name + "_val.pkl", processed_data_dir=ds_name + "_val")

    torch.save(tg_ds_train, raw_base_path + ds_name + "_train_tg_ds.pkl")
    torch.save(tg_ds_val, raw_base_path + ds_name + "_val_tg_ds.pkl")
    tg_ds_train = torch.load(raw_base_path + ds_name + "_train_tg_ds.pkl")
    tg_ds_val = torch.load(raw_base_path + ds_name + "_val_tg_ds.pkl")


    tg_dl_train = DataListLoader(tg_ds_train, shuffle=True, batch_size=4)
    tg_dl_val = DataListLoader(tg_ds_val, shuffle=True, batch_size=3)


  