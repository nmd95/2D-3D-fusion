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
from elbit_ds import Elbit_DS, torch_geometric_elbit_ds, Elbit_DS_multi, elbit_ds_lit
from models import PointNet_Graph_Vanilla_Parallel_lit, PointNet_Graph_MidFusion_NoAttention_Parallel_lit

from pytorch_lightning.metrics import Accuracy, IoU
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import (LightningDataModule, LightningModule, Trainer, seed_everything)
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import TensorBoardLogger
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from typing import Optional



if __name__ == '__main__':


    seed_everything(42)


    # dataset_name = "simple_no_fps"
    # dataset_name = "fusion_no_fps"
    dataset_name = "fusion_with_imgs_sample_factor_4"

    data_dir_path = "/mnt/storage/datasets/elbit_pointcloud_segmentation/tg_datasets/raw"
    logging_path= "/mnt/storage/datasets/elbit_pointcloud_segmentation/logging/Tensor_Board/logs"
    weights_path = "/mnt/storage/datasets/elbit_pointcloud_segmentation/logging/Tensor_Board/weights"
    ds_lit = elbit_ds_lit(data_dir=data_dir_path, ds_name=dataset_name, batch_size=16)

    # model = PointNet_Graph_Vanilla_Parallel_lit()
    model = PointNet_Graph_MidFusion_NoAttention_Parallel_lit()

    checkpoint_callback = ModelCheckpoint(monitor='val_iou', save_top_k=1, every_n_epochs=80)
    logger = TensorBoardLogger(save_dir=logging_path, name=dataset_name)

    trainer = Trainer(gpus=8, accelerator='ddp', max_epochs=1000,
                      callbacks=[checkpoint_callback], profiler='simple', auto_scale_batch_size=True, check_val_every_n_epoch=80, weights_save_path=weights_path, logger=logger)

    trainer.fit(model, datamodule=ds_lit)
    
    trainer.test()




