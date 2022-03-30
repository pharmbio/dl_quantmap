import numpy as np

from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs

from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable



# given list of labels, get counts for each label
def get_cluster_count_from_label(y_count):
    cluster_count = {}
    for y in y_count:
        if y not in cluster_count:
            cluster_count[y] = 1
        else:
            cluster_count[y] +=1
    return (cluster_count)


# Returns fingerprint for the input smiles
# smiles = smiles molecule
# ftype = fingerprint type (morgan,topological,MACCS,atompairs)
# radius = radius to be considered for morgan fingerprint
# bits = output bit vector size
# return_as_fp = returns as fingerprint (not as bit vector)
def smiles_fingerprint(smiles,ftype,radius=None,bits=2048,return_as_fp=False):
    try:
        m1 = Chem.MolFromSmiles(smiles)
    except:
        print (smiles)
    
    if ftype == "morgan":
        try:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(m1,radius,nBits=bits)
        except:
            print (smiles)
    if ftype == "topological":
        fp1 = Chem.RDKFingerprint(m1)
        
    if ftype == "MACCS":
        fp1 = MACCSkeys.GenMACCSKeys(m1)
        
    if ftype == "atompairs":
        fp1 = Pairs.GetAtomPairFingerprint(m1)
        
    if return_as_fp:
        return (fp1)
    else:
        bits = fp1.ToBitString()
        return (bits)
    

# Returns fingerprint list,label list for the input dict {smiles:label}
# smiles_label = {smiles:label} input dict
# ftype = fingerprint type (morgan,topological,MACCS,atompairs)
# radius = radius to be considered for morgan fingerprint
# bits = output bit vector size
# return_as_fp = returns as fingerprint (not as bit vector)
def convert_df_to_dict(input_df):
    df_dict = input_df.to_dict(orient='records')
    return  {entry["Smiles"]:entry["Label"] for entry in df_dict}

def smiles_to_fp(smiles_label,ftype,radius=None,bits=2048,return_as_fp=False):
    if type(smiles_label) != dict:
        smiles_label = convert_df_to_dict(smiles_label)
    
    x = []
    y= []
    
    for smiles in smiles_label:
        x.append([int(digit) for digit in smiles_fingerprint(smiles,ftype,radius,bits,return_as_fp)])
        y.append(int(smiles_label[smiles]))
        
    return (x,y)


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
    
            #accuracy.append(get_accuracy(output_val,yval))

            loss_val = criterion(output_val, yvalc)
            total_loss.append(loss_val.item())
            
            softmax = torch.exp(output_val.float())
            prob = softmax.cpu().detach().numpy()
            predictions = np.argmax(prob, axis=1)
            accuracy.append(accuracy_score(yval, predictions))
            
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

            loss_val = criterion(output_val, yvalc)

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
        
        loss_train = criterion(output_train, ybc)
        
        accuracy.append(get_accuracy(output_train,yb))
        
        loss_train.backward()
        optimizer.step()
        
        total_loss.append(loss_train.item())
    
    return (sum(total_loss)/(i+1),sum(accuracy)/(len(accuracy)))
