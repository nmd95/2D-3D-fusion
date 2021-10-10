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
from torchmetrics import IoU




from model_utils import knn_dgl_to_pg, random_subsample
from elbit_ds import get_pg_dl
from models import PointNet_Graph_Naive_RGB_Parallel
from elbit_ds import Elbit_DS, torch_geometric_elbit_ds, torch_geometric_elbit_ds_mid_fusion

def train_mid_fusion(model, train_dl, val_dl, epochs=3):
 
  model.train()
  
  optimizer = torch.optim.Adam(params=model.parameters())
  criterion = torch.nn.CrossEntropyLoss()
  losses = []

  val_every = 50

  for e in range(epochs):

    for step, sample in enumerate(tqdm(train_dl)):

      sample = (sample[0].squeeze().cuda(), sample[1].squeeze().cuda())

      if (sample[0].shape[0] < 2) or (len(sample[0].shape) < 2):
        print("skipping sample")
        continue
      
      print(sample[0].shape)
      print(sample[1].shape)
      # if step % val_every == (val_every - 1):
      #   print("\n val-results:", eval(model=model, eval_dl=val_dl).item())
      #   model.train()
      
      
      x_labels = sample[0][: , 3:4].long()

      # index = torch.from_numpy(np.random.choice((x_points_batch.shape[1]-1), int(x_points_batch.shape[1]/5.0))).long().cuda() # temporary - to be changed !  


      # print("x_labels", x_labels, x_labels.shape)
      y = model(sample)
      # print("\n y:", y, y.shape)
      # optimizer.zero_grad()
      model.zero_grad()
      loss = criterion(y.squeeze(), x_labels.squeeze())
      losses.append(loss.item())
      print("\n loss:", losses[-1])

      loss.backward()

      optimizer.step()
  
  return losses

def train_eval(model, train_dl, val_dl, epochs:int, log_file_path:str, save_weights_path:str): #def train_eval(train_dl, val_dl, epochs:int):
  
  eval_every = 60

  f = open(log_file_path, "a")

  optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #model.parameters- the weights of the model,  optim.Adam- doing gradiant descent on the weights of the model
   
  criterion = torch.nn.CrossEntropyLoss() #define lose function

  for epoch in tqdm(range(epochs)): 
      

    train_loss, train_passes = train_epoch(model=model, loader=train_dl, optimizer=optimizer, criterion=criterion) # train passes- for debuging how many samples were ignored to check that we don't ignore too much samples, sample= the image with it's point cloud
    train_stats = f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}'
    print(train_stats)
    f.write(train_stats)

    f.flush()

    if (epoch + 1) % eval_every == 0:
      test_acc, mean_iou, per_class_iou = test(model=model, loader=val_dl) #test_acc- ; mean_Iou- true positive/(true positive + false positive + false negative)
      test_stats = f'Epoch: {epoch:02d}, Test Accuracy: {test_acc:.4f}, Mean-IOU: {mean_iou:.4f}'
      print(test_stats)
      f.write(test_stats)
      print("\n per class iou: \n", per_class_iou)
      f.write("\n per class iou: \n")
      f.write(per_class_iou)
      f.flush()
      torch.save(model.state_dict(), save_weights_path + "/epoch_" + str(epoch) + ".pt")



  f.close()

def train_epoch(model, loader, optimizer, criterion):

    
    model.train()
    
    total_loss = 0
    train_passes = 0

    for data_list in tqdm(loader): # for data in tqdm(loader):
        # data = random_subsample(data, downsampling_factor=1.2).to(model.device)
        
        skip = 0
        for data in data_list:
          if ( data.pos.shape[0] < 2) :
            # print("skip!!!!!")
            skip = 1
            break
        if skip == 1:
          continue

        optimizer.zero_grad() 
        
        # logits = model(data.pos, data.x, data.batch)  
        logits = model(data_list)  
        # print('Outside Model: num points: {}'.format(logits.size(0)))
        y = torch.cat([data.y.squeeze().long() for data in data_list]).to(logits.device)
        loss = criterion(logits, y)
        # loss = criterion(logits, data.y.squeeze().long())
        loss.backward() #calculating backpropogation
        optimizer.step() #updating the weights by GD
        total_loss += loss.item() # * data.num_graphs

        # if steps % 2000 == 0:
        #   print(f'Current Training Loss: {loss.item():.4f}')

    # return total_loss / len(loader.dataset), train_passes
    return total_loss, train_passes



def test(model, loader):

  num_classes = 5
  
  def per_class_iou(confmat):
      values = confmat
      s = " "
      for j in range(0, num_classes):
        iou = values[j, j] / (values[j, : ].sum() + values[:,j].sum()- values[j, j])
        s += "\n IoU for class " + str(j) + " : " + str(iou.item())
      return s
  model.eval()
  with torch.no_grad():

    # def per_class_iou(IOU):
    #   values = np.array(IOU.get_weights()).reshape(14, 14)
    #   s = " "
    #   for j in range(0, 14):
    #     iou = values[j, j] / (values[j, : ].sum() + values[:,j].sum()- values[j, j])
    #     s += "\n IoU for class " + str(j) + " : " + str(iou)
    #   return s

    # # calculates f1-score, recall, precision for each class (from confusion-matrix:cm)
    # def cm_report(cm):
    #   scores = [(cm[k,k]/cm[k, :].sum(), cm[k,k]/cm[:, k].sum(), 2*((cm[k,k]/cm[:, k].sum() * cm[k,k]/cm[k, :].sum())/(cm[k,k]/cm[:, k].sum() + cm[k,k]/cm[k, :].sum())), cm[k, :].sum(), cm[:, k].sum()) for k in range(0,14)]
    #   report = {}
    #   for j in range(0, 13):
    #     temp = {'precision': scores[j][0], 'recall': scores[j][1], 'f1-score': scores[j][2], 'support_preds': scores[j][3], 'support_gt': scores[j][4]}
    #     s = "label " + str(j)
    #     report[s] = temp
    #   return report

    # IOU = MeanIoU(num_classes=num_classes)
    IOU = IoU(num_classes=num_classes).to("cuda:0")

    total_correct = 0
    total_points = 0
    test_passes = 0
    
    # confusion_matrix = torch.zeros((14, 14)).to(model.device) # colums count true labels, rows: the predictions
    
    # for data in tqdm(loader): 
        
    #     data = data.to('cuda:0')
    #     if data.num_nodes < 1:
    #       test_passes += 1
    #       continue
    for data_list in tqdm(loader): 

      for data in data_list:
        skip = 0
        for data in data_list:
          if ( data.pos.shape[0] < 2) :
            # print("skip!!!!!")
            skip = 1
            break
        if skip == 1:
          continue
        
      
        
        # print('Outside Model: num graphs: {}'.format(logits.size(0)))

        logits = model(data_list)
        
        y = torch.cat([data.y.squeeze().long() for data in data_list]).to(logits.device)
        
        pred = logits.argmax(dim=-1)

        # total_correct += ((pred == y).sum()) # total_correct += ((pred == data.y.squeeze()).sum())
        # total_points += y.shape[0] #total_points += data.y.shape[0]

        # confusion_matrix[pred.long(), data.y.squeeze().long()] += 1.0
        # IOU.update_state(y.cpu().numpy(), pred.cpu().numpy()) # IOU.update_state(data.y.squeeze().cpu().numpy(), pred.cpu().numpy())
       
        IOU.update(pred, y)
    
    # print(cm_report(confusion_matrix))


    
    
    
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix.transpose(0, 1).cpu().numpy(), display_labels=np.asarray([i for i in range(0,14)]))
    # disp.plot() 

    # return total_correct / total_points, IOU.result(), per_class_iou(IOU)
    # return total_correct / total_points, IOU.compute(), per_class_iou(IOU.confmat)
  return 0.0, IOU.compute(), per_class_iou(IOU.confmat)
