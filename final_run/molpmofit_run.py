import os
import gc
import glob
import sys
import random
import string
import tqdm
import json
import time
import sqlite3
import warnings
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit import RDLogger

from SmilesPE.pretokenizer import atomwise_tokenizer
from SmilesPE.pretokenizer import kmer_tokenizer
from SmilesPE.spe2vec import Corpus

from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

from multiprocessing import Pool

from fastai import *
from fastai.text import *
#from utils import *
import torch

sys.path.append('/scratch-shared/akshai/Publication/supp_scripts/')
import supp_utils as su

#torch.cuda.set_device(0) #change to 0 if you only has one GPU
# set gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device,torch.cuda.is_available()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
# To remove rdkit warning
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

parameter_filename = "parameters.json" 

parameter_file = open(parameter_filename)
parameters = json.load(parameter_file)
parameter_file.close()

# User inputs
input_file_train = parameters["input_file_train"] # input file
input_file_test = parameters["input_file_test"] # input file

trial = parameters["trial"] # setting False saves the output files else not saved

if not trial:
    run_folder = parameters["run_folder"]

gpu_id = int(parameters["gpu_id"])
if gpu_id != None:
    device = "cuda:" + str(gpu_id)
else:
    gpu_id = 0

torch.cuda.set_device(device)
# User inputs
trial = parameters["trial"] # setting False saves the output files else not saved

# Removing data with lower distribution
#enable_label_cutoff = parameters["label_cutoff"]["enable_label_cutoff"]
lower_label_count_cutoff = int(parameters["label_cutoff"]["lower_label_count_cutoff"])
upper_label_count_cutoff = int(parameters["label_cutoff"]["upper_label_count_cutoff"])

k_fold_value = int(parameters["k_fold_value"]) # Number of folds

test_set_percentage = float(parameters["test_set_percentage"])

label_wise_augmentation = parameters["augmentation"]["label_wise_augmentation"]
number_of_augmentation = int(parameters["augmentation"]["number_of_augmentation"])
iteration = int(parameters["augmentation"]["iteration"])

tokenization = parameters["tokens"]["tokenization"] # options are SPE,atomwise,vocab_file
if tokenization == "SPE":
    spe_token_path = parameters["tokens"]["spe_token_path"]
else:
    spe_token_path = ""
    
#####################
# Network parameters#
#####################
load_model = parameters["pretrained_model"]["load_model"]
#if load_model is True set the path for pretrained_model_path
pretrained_model_path = parameters["pretrained_model"]["pretrained_model_path"]
pretraining_new_wt = parameters["pretrained_model"]["pretraining_new_wt"]
pretraining_new_vocab = parameters["pretrained_model"]["pretraining_new_vocab"]

epochs = int(parameters["network_parameters"]["epochs"])
batch_size = int(parameters["network_parameters"]["batch_size"])
learning_rate = float(parameters["network_parameters"]["learning_rate"])
enable_class_weight = parameters["network_parameters"]["enable_class_weight"]

Number_of_workers = int(parameters["Number_of_workers"])


##################
### Do not edit###
##################
os.system("mkdir " + str(run_folder))

atomwise_tokenization = False
train_SPE = False

if tokenization == "SPE":
    train_SPE = True
else:
    atomwise_tokenization = True

if not trial:
    network_parameter_output = open(str(run_folder) + "/network_parameters.txt","w",1)
    for parameter in parameters:
        network_parameter_output.write(str(parameter) + " = " + str(parameters[parameter]) + "\n")
        
smiles_label_test = {line.split()[0]:line.split()[1] for line in open(input_file_test,"r").readlines()}
smiles_label_test = dict(sorted(smiles_label_test.items(), key=lambda item: item[1]))

smiles_label_train = {line.split()[0]:line.split()[1] for line in open(input_file_train,"r").readlines()}
smiles_label_train = dict(sorted(smiles_label_train.items(), key=lambda item: item[1]))

train_valid_df = su.dict_to_label(smiles_label_train)
train_valid_df = train_valid_df
test_df = su.dict_to_label(smiles_label_test)
test_df = test_df.sample(frac=1).reset_index(drop=True)

data_path = Path(run_folder)
#name = 'classification_new'
path = data_path
path.mkdir(exist_ok=True, parents=True)

gc.collect()
torch.cuda.empty_cache()

def get_accuracy(yhat,y):
    softmax = torch.exp(yhat.float())
    prob = softmax.cpu().detach().numpy()
    predictions = np.argmax(prob, axis=1)
    y_truth = y.cpu().detach().numpy()
    accuracy_check = (y_truth==predictions)
    count = np.count_nonzero(accuracy_check)
    accuracy = (count/len(accuracy_check))
    return accuracy

if not trial:
    log_file = open(str(run_folder) + "/model_log.txt","w",1)

train_df,valid_df,_ = su.split_data_with_label(smiles_label_train,train_percentage=1-test_set_percentage,valid_percentage=test_set_percentage)
print ("Split contains=",len(train_df),len(valid_df))
log_file.write("Split contains train = " + str(len(train_df)) + "\n")
log_file.write("Split contains valid = " + str(len(valid_df)) + "\n")
log_file.write("Split contains test = " + str(len(test_df)) + "\n")
train_set = str(run_folder) + "/train_set.txt"
valid_set = str(run_folder) +"/valid_set.txt"
if not os.path.isfile(train_set) and not os.path.isfile(valid_set):
    train_df.to_csv(train_set,sep=" ",header=False,index=False)
    valid_df.to_csv(valid_set,sep=" ",header=False,index=False)
else:
    train_df = pd.read_csv(train_set,sep=" ",names=["Smiles", "Label"])
    valid_df = pd.read_csv(valid_set,sep=" ",names=["Smiles", "Label"])
    print ("ERROR: Cannot write, file already present\n\n")
    print ("Using already present files")
    print (len(train_df),len(valid_df))
    log_file.write("ERROR: Cannot write, file already present\n\n")
    log_file.write("Using already present files")
    log_file.write("Split contains train = " + str(len(train_df)) + "\n")
    log_file.write("Split contains valid = " + str(len(valid_df)) + "\n")
    log_file.write("Split contains test = " + str(len(test_df)) + "\n")
    

        
# calculate class_weight
if enable_class_weight:
    class_weight = torch.FloatTensor(su.get_class_weight(train_df)).cuda()
    if not trial:
        log_file.write("Class weight for loss (balancing weights)= " + str(class_weight) + "\n")
        
if not trial:
    log_file.write("Class distribution before augmentation\n")
    log_file.write("Train data\n")
    log_file.write(str(train_df.groupby('Label').count()) + "\n")
    log_file.write("Valid data\n")
    log_file.write(str(valid_df.groupby('Label').count()) + "\n")
    log_file.write("Test data\n")
    log_file.write(str(test_df.groupby('Label').count()) + "\n")

# Data augmentation
if number_of_augmentation > 0:
        
    train_aug_path = "data/classification/train_data_aug_canonical_smiles.csv"
    valid_aug_path = "data/classification/valid_data_aug_canonical_smiles.csv"
    test_aug_path = "data/classification/test_data_aug_canonical_smiles.csv"
    
    if not os.path.isfile(train_aug_path) or  not os.path.isfile(valid_aug_path) or not os.path.isfile(test_aug_path):
        if label_wise_augmentation:

            train_augmentation_list = su.get_augmentation_list(train_df,number_of_augmentation)
            number_of_augmentation_train = train_augmentation_list

            valid_augmentation_list = su.get_augmentation_list(valid_df,number_of_augmentation)
            number_of_augmentation_valid = valid_augmentation_list

            test_augmentation_list = su.get_augmentation_list(test_df,number_of_augmentation)
            number_of_augmentation_test = test_augmentation_list

        else:   
            number_of_augmentation_train = number_of_augmentation
            #number_of_augmentation_valid = number_of_augmentation
            #if fold == 0:
            #    number_of_augmentation_test = number_of_augmentation

        train_data = su.smiles_augmentation(train_df,
                                                N_rounds=number_of_augmentation_train,
                                                iteration=iteration,
                                                data_set_type="train_data",
                                                Number_of_workers=Number_of_workers)     

        valid_data = su.smiles_augmentation(valid_df,
                                                N_rounds=0,
                                                iteration=iteration,
                                                data_set_type="valid_data",
                                                Number_of_workers=Number_of_workers)

        test_data = su.smiles_augmentation(test_df,
                                                N_rounds=0,
                                                iteration=iteration,
                                                data_set_type="test_data",
                                                Number_of_workers=Number_of_workers)
    else:
        train_data = pd.read_csv(train_aug_path,sep=",")
        valid_data = pd.read_csv(valid_aug_path,sep=",")
        test_data = pd.read_csv(test_aug_path,sep=",")
        print ("ERROR: Cannot augment, file already present\n\n")
        print ("Using already present files")
        print ("Train_data Valid_data Test_data")
        print (len(train_data),len(valid_data),len(test_data))
        log_file.write("ERROR: Cannot augment, file already present\n\n")
        log_file.write("Using already present files\n")
        log_file.write("Split contains train = " + str(len(train_data)) + "\n")
        log_file.write("Split contains valid = " + str(len(valid_data)) + "\n")
        log_file.write("Split contains test = " + str(len(test_data)) + "\n")
    # calculate class_weight
        
    if enable_class_weight:
        class_weight = torch.FloatTensor(su.get_class_weight(train_data)).cuda()
        if not trial:
            log_file.write("Class weight for loss (balancing weights) after augmentation= " + str(class_weight) + "\n")
        
    classes = list(set(train_data["Label"].tolist()))
    classes = list(sorted(list(map(int,classes))))
        
    #valid_data = valid_df
    #test_data = test_df
    
        
    if not trial:
        log_file.write("number of augmentation = " + str(number_of_augmentation) + "\n")
        log_file.write("Class distribution after augmentation\n")
        log_file.write("Train data\n")
        log_file.write(str(train_data.groupby('Label').count()) + "\n")
        log_file.write("Valid data\n")
        log_file.write(str(valid_data.groupby('Label').count()) + "\n")
        log_file.write("Test data\n")
        log_file.write(str(valid_data.groupby('Label').count()) + "\n")
else:
    train_data = train_df
    valid_data = valid_df
    test_data = test_df
    train_data['Label'] = train_data['Label'].astype(str)
    valid_data['Label'] = valid_data['Label'].astype(str)
    test_data['Label'] = test_data['Label'].astype(str)
    classes = list(set(train_data["Label"].tolist()))
    classes = list(map(str,sorted(list(map(int,classes)))))
    
    
if tokenization == "SPE":
    MolTokenizer = su.molpmofit.MolTokenizer_spe_sos_eos
else:
    MolTokenizer = su.molpmofit.MolTokenizer_atomwise_sos_eos

tok = Tokenizer(partial(MolTokenizer,token_path=spe_token_path), n_cpus=Number_of_workers, pre_rules=[], post_rules=[])
    
qsar_vocab = TextLMDataBunch.from_df(path, train_data, valid_data, bs=batch_size, tokenizer=tok,chunksize=50000, text_cols=0,label_cols=1, max_vocab=60000, include_bos=False)
    
pretrained_model_path = Path(pretrained_model_path)

pretrained_fnames = [pretraining_new_wt, pretraining_new_vocab]
fnames = [pretrained_model_path/f'{fn}.{ext}' for fn,ext in zip(pretrained_fnames, ['pth', 'pkl'])]

lm_learner = language_model_learner(qsar_vocab, AWD_LSTM, drop_mult=1.0)
lm_learner = lm_learner.load_pretrained(*fnames)
lm_learner.freeze()
lm_learner.save_encoder(f'lm_encoder')
    

data_clas = TextClasDataBunch.from_df(path, train_data, valid_data, bs=batch_size, tokenizer=tok, 
                                              chunksize=50000, text_cols='Smiles',label_cols='Label', 
                                              vocab=qsar_vocab.vocab, max_vocab=60000, include_bos=False,classes=classes)
    
if enable_class_weight:
    cls_learner = text_classifier_learner(data_clas, AWD_LSTM, pretrained=False, drop_mult=0.2,loss_func=nn.CrossEntropyLoss(weight=class_weight))
else:
    cls_learner = text_classifier_learner(data_clas, AWD_LSTM, pretrained=False, drop_mult=0.2)#,loss_func=nn.CrossEntropyLoss(weight=class_weight))
cls_learner.load_encoder(f'lm_encoder',device=device)
    
cls_learner.loss_func = nn.CrossEntropyLoss()
    
cls_learner.freeze()
cls_learner.fit_one_cycle(4, 3e-3, moms=(0.8,0.7))
cls_learner.freeze_to(-2)
cls_learner.fit_one_cycle(4, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))
cls_learner.freeze_to(-3)
cls_learner.fit_one_cycle(4, slice(5e-4/(2.6**4),5e-4), moms=(0.8,0.7))
cls_learner.unfreeze()
cls_learner.fit_one_cycle(6, slice(5e-5/(2.6**4),5e-5), moms=(0.8,0.7))
    
split_type = ""
split_id = "model"
cls_learner.save(f'{split_type}_{split_id}_clas')
gc.collect()
torch.cuda.empty_cache()
    
train_data_clas = TextClasDataBunch.from_df(path, train_data, train_data, bs=batch_size, tokenizer=tok, 
                              chunksize=50000, text_cols='Smiles',label_cols='Label', vocab=qsar_vocab.vocab, max_vocab=60000,
                                              include_bos=False,classes=classes)
learner = text_classifier_learner(train_data_clas, AWD_LSTM, pretrained=False, drop_mult=0.2)
learner.load(f'{split_type}_{split_id}_clas', purge=False);
pred,lbl,loss = learner.get_preds(with_loss=True,ordered=True)
    
accuracy = str(get_accuracy(pred,lbl))
loss = str(sum(loss)/len(loss))
softmax = torch.exp(pred.float())
prob = softmax.cpu().detach().numpy()
predictions = np.argmax(prob, axis=1)
y_truth = lbl.cpu().detach().numpy()
target_names = ["class " + str(entry) for entry in range(len(set(y_truth)))]
report = classification_report(y_truth, predictions, target_names=target_names,digits=7)
log_file.write("\n\n\nTrain data : Accu-" + str(accuracy) + "\tLoss-" + str(loss) + "\n")
log_file.write("Train data report \n-" + str(report) + "\n\n\n\n\n")
gc.collect()
torch.cuda.empty_cache()
    
valid_data_clas = TextClasDataBunch.from_df(path, train_data, valid_data, bs=batch_size, tokenizer=tok, 
                              chunksize=50000, text_cols='Smiles',label_cols='Label', vocab=qsar_vocab.vocab, max_vocab=60000,
                                              include_bos=False,classes=classes)
learner = text_classifier_learner(valid_data_clas, AWD_LSTM, pretrained=False, drop_mult=0.2)
learner.load(f'{split_type}_{split_id}_clas', purge=False);
pred,lbl,loss = learner.get_preds(with_loss=True,ordered=True)
    
accuracy = str(get_accuracy(pred,lbl))
loss = str(sum(loss)/len(loss))
softmax = torch.exp(pred.float())
prob = softmax.cpu().detach().numpy()
predictions = np.argmax(prob, axis=1)
y_truth = lbl.cpu().detach().numpy()
target_names = ["class " + str(entry) for entry in range(len(set(y_truth)))]
report = classification_report(y_truth, predictions, target_names=target_names,digits=7)
log_file.write("\n\n\nValid data : Accu-" + str(accuracy) + "\tLoss-" + str(loss) + "\n")
log_file.write("Valid data report \n-" + str(report) + "\n\n\n\n\n")
gc.collect()
torch.cuda.empty_cache()
    
test_data_clas = TextClasDataBunch.from_df(path, train_data, test_data, bs=batch_size, tokenizer=tok, 
                              chunksize=50000, text_cols='Smiles',label_cols='Label', vocab=qsar_vocab.vocab, max_vocab=60000,
                                              include_bos=False,classes=classes)
learner = text_classifier_learner(test_data_clas, AWD_LSTM, pretrained=False, drop_mult=0.2)
learner.load(f'{split_type}_{split_id}_clas', purge=False);
pred,lbl,loss = learner.get_preds(with_loss=True,ordered=True)

accuracy = str(get_accuracy(pred,lbl))
loss = str(sum(loss)/len(loss))
softmax = torch.exp(pred.float())
prob = softmax.cpu().detach().numpy()
predictions = np.argmax(prob, axis=1)
y_truth = lbl.cpu().detach().numpy()
target_names = ["class " + str(entry) for entry in range(len(set(y_truth)))]
report = classification_report(y_truth, predictions, target_names=target_names,digits=7)
log_file.write("\n\n\nTest data : Accu-" + str(accuracy) + "\tLoss-" + str(loss) + "\n")
log_file.write("Test data report \n-" + str(report) + "\n\n\n\n\n")
    
lm_learner.destroy()
cls_learner.destroy()
learner.destroy()
gc.collect()
torch.cuda.empty_cache()