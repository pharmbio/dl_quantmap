import numpy as np

from sklearn.metrics import accuracy_score

from rdkit import Chem

from feature import *
import SCFPfunctions as Mf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# converts input file of smiles label to feature matrix
# file = input filename
# lensize = x-axis length
# atomsize = y-axis length (determines the sequence length cutoff)
# returns x and y
# Function adapted from Hirohara et al
def make_grid(file,lensize=42,atomsize=400):
    xp=np
    #print('Loading smiles: ', file)
    smi = Chem.SmilesMolSupplier(file,delimiter=' ',titleLine=False,sanitize=False)
    mols = [mol for mol in smi if mol is not None]
    
    F_list, T_list = [],[]
    for mol in mols:
        if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > atomsize: print("too long mol was ignored")
        else:
            F_list.append(mol_to_feature(mol,-1,atomsize))
            T_list.append(mol.GetProp('_Name'))
    Mf.random_list(F_list)
    Mf.random_list(T_list)

    data_t = xp.asarray(T_list, dtype=xp.int32).reshape(-1,1)
    data_f = xp.asarray(F_list, dtype=xp.float32).reshape(-1,1,atomsize,lensize)

    dataset = (data_f,data_t)
    #print(data_t.shape, data_f.shape)
    
    return (data_f,data_t)


# Get accuracy for data given yhat and y
# yhat = predictions
# y = ground truth
def get_accuracy(yhat,y):
    softmax = torch.exp(yhat.float())
    prob = list(softmax.cpu().detach().numpy())
    predictions = (np.argmax(prob, axis=1))
    return accuracy_score(y, predictions)


# given the model,criterion,dataloader,device returns loss,accuracy,prediction_list
# prediction_list = list of [[ground_truth],[predictions]]
def test(model,criterion,val_dl,device="cpu",get_prediction_list=True):
    model.eval()
    total_loss = []
    accuracy = []
    
    real_and_predictions = [[],[]]
    
    with torch.no_grad():  
        for i, (xval,yval) in enumerate(val_dl):
            xval = xval.to(device)
            yvalc = yval.to(device)

            output_val = model(xval.float())

            accuracy.append(get_accuracy(output_val,yval))

            loss_val = criterion(output_val, yvalc.long())
            total_loss.append(loss_val.item())
            
            softmax = torch.exp(output_val.float())
            prob = softmax.cpu().detach().numpy()
            predictions = np.argmax(prob, axis=1)
            y_truth = yvalc.cpu().detach().numpy()
            
            if get_prediction_list:
                real_and_predictions[0].extend(y_truth)
                real_and_predictions[1].extend(predictions)
            else:
                real_and_predictions = None
            
        return (sum(total_loss)/(i+1),sum(accuracy)/(len(accuracy)),real_and_predictions)


# find loss and accuracy for validation set
# model = model as input
# criterion = criterion as input
# val_dl = validation dataloader
# device = device for calculation
def validate(model,criterion,val_dl,device="cpu"):
    model.eval()
    total_loss = []
    accuracy = []
    
    with torch.no_grad():  
        for i, (xval,yval) in enumerate(val_dl):
            xval = xval.to(device)
            yvalc = yval.to(device)

            output_val = model(xval.float())

            accuracy.append(get_accuracy(output_val,yval))

            loss_val = criterion(output_val, yvalc.long())

            total_loss.append(loss_val.item())
        
        return (sum(total_loss)/(i+1),sum(accuracy)/(len(accuracy)))


# train the model
# find loss and accuracy for train set
# model = model as input
# criterion = criterion as input
# optimizer = optimizer as input
# train_dl = train dataloader
# device = device for calculation
def train(model,criterion,optimizer,train_dl,device="cpu"):
    model.train()
    total_loss = []
    accuracy = []
    for i, (xb,yb) in enumerate(train_dl):
        xb = xb.to(device)
        ybc = yb.to(device)
        
        optimizer.zero_grad()
        
        output_train = model(xb.float())
        
        loss_train = criterion(output_train, ybc.long())
        
        accuracy.append(get_accuracy(output_train,yb))
        
        loss_train.backward()
        optimizer.step()
        
        total_loss.append(loss_train.item())
    
    return (sum(total_loss)/(i+1),sum(accuracy)/(len(accuracy)))