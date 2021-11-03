import numpy as np
import math
import random


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# receives {smiles:label} and get the k_fold split based on piece_count (current_kfold)
# smiles_label = input {smiles:label}
# k_fold = input k_fold number
# piece_count = current piece count (increments after each use)
# shuffle_output = shuffle the output train and test dict
# returns output train and test dict of {smiles:label} format with incremented piece_count
def convert_df_to_dict(input_df):
    df_dict = input_df.to_dict(orient='records')
    return  {entry["Smiles"]:entry["Label"] for entry in df_dict}

def get_K_fold_cv_data(smiles_label,k_fold,piece_count=1,shuffle_output=True):
    
    if type(smiles_label) != dict:
        smiles_label = convert_df_to_dict(smiles_label)
    
    labels = sorted(set(list(smiles_label.values())))
    
    train_smiles_label = {}
    test_smiles_label = {}
    
    for label in labels:
        current_set_of_smiles = []
        
        for smiles in smiles_label:
            if smiles_label[smiles] == label:
                current_set_of_smiles.append(smiles)
                
        x,y = current_set_of_smiles,np.array([label for entry in current_set_of_smiles])
    
        total_length = y.shape[0]
        split_count = math.ceil(total_length/k_fold)

        previous_split = (piece_count - 1) * split_count
        current_split = piece_count * split_count
        
        x_test = x[previous_split:current_split]
        y_test = y[previous_split:current_split]

        exclude_list = [entry  for entry in range(current_split) if entry >= previous_split and entry < total_length]
        x_train = np.delete(x, exclude_list,axis=0)
        y_train = np.delete(y, exclude_list,axis=0)
        
        for i,smiles in enumerate(x_train):
            train_smiles_label[smiles] = y_train[i]
        
        for i,smiles in enumerate(x_test):
            test_smiles_label[smiles] = y_test[i]

    if shuffle_output:
        l_train = list(train_smiles_label.items())
        random.shuffle(l_train)
        train_smiles_label = dict(l_train)
        
        l_test = list(test_smiles_label.items())
        random.shuffle(l_test)
        test_smiles_label = dict(l_test)
        
        
    piece_count += 1
    
    return (train_smiles_label,test_smiles_label,piece_count)


# Create dataloader for train and test files
# x_train,y_train = x and y train
# x_test,y_test = x and y test
# batch_size = batch size for dataloader
def get_dataloader(x,y,batch_size=32):
    
    # train
    if type(x) == list:
        x = np.array(x)
    if type(y) == list:
        y = np.array(y)
        
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    
    dataset = torch.utils.data.TensorDataset(xt, yt)
    
    data_loader = torch.utils.data.DataLoader(dataset, 
                                              batch_size=batch_size)
    
    return (data_loader)