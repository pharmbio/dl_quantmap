import os
import time
import gc
from multiprocessing import Pool
from functools import partial
import numpy as np
import glob
import codecs
import requests

import itertools

from rdkit import Chem

import torch

from SmilesPE.pretokenizer import atomwise_tokenizer
from SmilesPE.pretokenizer import kmer_tokenizer
from SmilesPE.learner import *
from SmilesPE.tokenizer import *

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import sklearn

import string
import tqdm
import pandas as pd

# loading custom functions
import DNN
try:
    import CNN
except:
    print ("Could not import custom script CNN")
import seq2seq
import molpmofit
import CV


# Write dict of {smiles:cid} to file with " " spacer
# or dataframe with column names "Smiles" and "Label"
# input_dict_df = input dict or dataframe
# filename = output_filename
def convert_df_to_dict(input_df):
    df_dict = input_df.to_dict(orient='records')  
    return  {entry["Smiles"]:entry["Label"] for entry in df_dict}

def write_cid_smiles_output(input_dict_df,filename):
    if type(input_dict_df) != dict:
        input_dict_df = convert_df_to_dict(input_dict_df)
        
    output_file = open(filename,"w")
    
    for smiles in input_dict_df:
        output_file.write(str(smiles) + " " + str(input_dict_df[smiles]) + "\n")
    
    output_file.close()


# plot confusion matrix and generates report of precision,recall and f1 score
# prediction_list = list of [[ground_truth],[predictions]]
# image_name = output filename for the image
def confustion_matrix(prediction_list,image_name=None,plot_cm = True):
    y_ground_truth,y_predicted =   prediction_list[0],prediction_list[1]
    target_names = ["class " + str(entry) for entry in range(len(set(y_ground_truth)))]
    cm = sklearn.metrics.confusion_matrix(y_ground_truth,y_predicted)
    
    if plot_cm:
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        cmap = plt.get_cmap('Blues')

        plt.imshow(cm, cmap=cmap, interpolation='nearest')

        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)


        thresh = cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        if image_name != None:
            plt.savefig(image_name)
    
    report = classification_report(y_ground_truth, y_predicted, target_names=target_names,digits=7)
    return (report)


# callback function to save the best model from the epoch
# current_epoch_values = current epoch values such as [train_f1,valid_f1]
# previous_epoch_values = previous epoch list of values = [train_f1_list,valid_f1_list]
# model = input model
# model_name = model name including path
# saves the model if it is better than any previous model
def callback_with_f1(current_epoch_values,previous_epoch_values,model,model_name=None):
    train_f1 =  current_epoch_values[0]
    val_f1 = current_epoch_values[1]
    
    train_f1_list = previous_epoch_values[0]
    val_f1_list = previous_epoch_values[1]
    
    max_f1_valid_index = val_f1_list.index(sorted(val_f1_list)[-1])
    if val_f1 > val_f1_list[max_f1_valid_index]:
        return True #save_model(model,model_name)
    elif val_f1 == val_f1_list[max_f1_valid_index]:
        if train_f1 > train_f1_list[max_f1_valid_index]:
            return True #save_model(model,model_name)
        else:
            pass
    else:
        pass
    
# callback function to save the best model from the epoch
# current_epoch_values = current epoch values such as [train_loss,train_accu,val_loss,val_accu]
# previous_epoch_values = previous epoch list of values = [train_loss_list,train_accu_list,val_loss_list,val_accu_list]
# model = input model
# model_name = model name including path
# saves the model if it is better than any previous model
def save_model(model,model_name):
    torch.save(model.state_dict(), model_name)
    return True

def callback(current_epoch_values,previous_epoch_values,model,model_name=None):
    train_loss =  current_epoch_values[0]
    train_accu = current_epoch_values[1]
    val_loss = current_epoch_values[2]
    val_accu = current_epoch_values[3]
    
    train_loss_list = previous_epoch_values[0]
    train_accu_list = previous_epoch_values[1]
    val_loss_list = previous_epoch_values[2]
    val_accu_list = previous_epoch_values[3]
    
    
    least_loss_valid_index = val_loss_list.index(sorted(val_loss_list)[0])
    if val_loss < val_loss_list[least_loss_valid_index]:
        return True #save_model(model,model_name)
        
    elif val_loss == val_loss_list[least_loss_valid_index]:
        if train_loss < train_loss_list[least_loss_valid_index]:
            return True #save_model(model,model_name)
                
        elif train_loss == train_loss_list[least_loss_valid_index]:
            if val_accu < val_accu_list[least_loss_valid_index]:
                return True #save_model(model,model_name)
                    
            elif val_accu == val_accu_list[least_loss_valid_index]:
                if train_accu < train_accu_list[least_loss_valid_index]:
                    return True #save_model(model,model_name)
                        
                else:
                     pass
            else:
                 pass
        else:
             pass
    else:
         pass


# get class weight for the input data
# smallest class/each class
# smallest class would be 1 and largest would be in fractions
# returns class weight for each class in order
def get_class_weight(input_data):
    if type(input_data) != dict:
        class_count_list = np.array([entry for entry in input_data.groupby('Label').count()["Smiles"]])
        class_weight = np.min(class_count_list)/class_count_list
    elif type(input_data) == dict:
        all_labels_list = list(input_data.values())
        class_count_list = np.array([all_labels_list.count(label) for label in sorted(set(all_labels_list))])
        class_weight = np.min(class_count_list)/class_count_list
    else:
        all_labels_list = input_data
        class_count_list = np.array([all_labels_list.count(label) for label in sorted(set(all_labels_list))])
        class_weight = np.min(class_count_list)/class_count_list
        
    return (class_weight)

# Remove multiple entries in a list of smiles
# input_list = input list of smiles
# Returns new list
def remove_duplicates_list(input_list):
    return list(set(input_list))

# Remove multiple entries in a file of smiles and label
# Returns new filename
def remove_duplicates_from_file(filename,label_present=True):
    clean_file = filename[:-4] + "_out.txt"
    open_file = open(filename,"r").readlines()
    output = open(clean_file,"w")
    finished_list = []
    if label_present:
        loop = tqdm.tqdm(open_file, total=len(open_file),leave=False)
        for entry in loop:
                if entry.split()[0] not in finished_list:
                    finished_list.append(entry.split()[0])
                    output.write(entry.split()[0] + " " + entry.split()[1] + "\n")
    else:
        finished_list = list(set(open_file))
        loop = tqdm.tqdm(finished_list, total=len(finished_list),leave=False)
        for entry in loop:
            output.write(entry + "\n")
    output.close()
    return clean_file


# sort {smiles:label} dict based on smiles for each label
def sort_smiles_label(smiles_label):
    labels = sorted(set(list(smiles_label.values())))
    
    output_smiles_label = {}
    
    for label in labels:
        current_set_of_smiles = []
        
        for smiles in smiles_label:
            if smiles_label[smiles] == label:
                current_set_of_smiles.append(smiles)
            
        sorted_list = sorted(current_set_of_smiles)
        
        for smiles in sorted_list:
            output_smiles_label[smiles] = label
            
    return (output_smiles_label)


# To get data within the cutoff (given the data and based on label count)
# returns {smiles:label} dict and chosen {label:count} dict
# selected labels are relabelled starting from "0"
# input_filename = input filename
# lower_label_count_cutoff = lower cutoff for the label
# upper_label_count_cutoff = upper cutoff for the label
def get_data_within_cutoff(input_filename,lower_label_count_cutoff=0,upper_label_count_cutoff=100000,sort=True,sanitize=False,canonical=False):
    open_file = open(input_filename,"r").readlines()
    
    # Find the  number of labels (count of each label)
    label_count_init = {}
    for entry in open_file:
        label = int(entry.split()[1])
        if label in label_count_init:
            label_count_init[label] += 1
        else:
            label_count_init[label] = 1
    
    # Select the label within the cutoff  (count of each label within cutoff)
    label_count = {}
    for entry in label_count_init:
        if label_count_init[entry] > lower_label_count_cutoff and label_count_init[entry] < upper_label_count_cutoff:
            label_count[entry] = label_count_init[entry]
    
    # Select only the smiles with labels within the cutoff
    smiles_label = {}
    allocated_label = []
    allowed_labels = sorted(label_count.keys())
    
    loop = tqdm.tqdm(open_file, total=len(open_file),leave=False)
    sanity_removal = 0
    for entry in loop:
        smiles = entry.split()[0]
        label = int(entry.split()[1])
        if label in allowed_labels:
            if label not in allocated_label:
                allocated_label.append(label)
                    
            if sanitize:
                if canonical:
                    smiles = sanitize_molecule(output_type="canonical",smiles=entry.split()[0])
                else:
                    smiles = sanitize_molecule(smiles=smiles)
                
            if smiles is None:
                sanity_removal += 1
            else:
                smiles_label[smiles] = allocated_label.index(label)
    
    if sanitize:
        print (str(sanity_removal) + " molecules removed after sanity check")
    print (str(len(smiles_label)) + "/" + str(len(open_file)) + " data points obtained")
    
    if sort:
        smiles_label = sort_smiles_label(smiles_label)
        return (smiles_label,label_count)
    else:
        return smiles_label,label_count
    

# Sanity check for single molecule 
# output_type=None (canonical - for canonical smiles output)
# smiles = Input smiles of the molecule
def sanitize_molecule(output_type=None,smiles=None):
    molecule = Chem.MolFromSmiles(smiles,sanitize=False)
    if molecule is None:
        return None
    else:
        try:
            Chem.SanitizeMol(molecule)
            if output_type == "canonical":
                return Chem.MolToSmiles(molecule)
            else:
                return smiles
        except:
            return None

        
# List or dataframe with header ("Smiles,Label") as input for sanity check
# output_type=None (canonical - for canonical smiles output)
# Number_of_workers(1) = to run in pool of threads
def sanity_check(df,output_type = None,Number_of_workers = 1):
    if type(df) == list:
        func = partial(sanitize_molecule,output_type)
        
        p = Pool(Number_of_workers)
        clean_smiles = list(tqdm.tqdm(p.imap(func, df), total=len(df),leave=False))
        clean_smiles_2 = [smiles for smiles in clean_smiles if smiles != None]
        p.close()
        
        return clean_smiles_2
    
    else:
        labels = []
        for label in df.groupby('Label'):
            labels.append(label[0])
        
        clean_smiles_list = []
        label_array = []
        
        for label in labels:
            
            canonical_smiles = df[df['Label'] == label]['Smiles'].to_list()
            
            func = partial(sanitize_molecule,output_type)
            
            p = Pool(Number_of_workers)
            clean_smiles = list(tqdm.tqdm(p.imap(func, canonical_smiles), total=len(canonical_smiles),leave=False))
            clean_smiles_2 = [smiles for smiles in clean_smiles if smiles != None]
            clean_smiles_list.extend(clean_smiles_2)
            p.close() 
            
            label_array.extend(label * np.ones(len(clean_smiles_2),dtype=int))
            
        output_df = pd.DataFrame(columns=["Smiles","Label"]) 
        output_df["Smiles"] = clean_smiles_list
        output_df["Label"]  = label_array
    
        return  output_df

    
# Randomize the atom order to get a new smiles string
def randomize_smiles(smiles,random_smiles=[],iteration=5):
    try:
        m = Chem.MolFromSmiles(smiles)
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        out_smiles = (Chem.MolToSmiles(nm, canonical=False, isomericSmiles=True, kekuleSmiles=False))
    except:
        return (False)
    
    if out_smiles not in random_smiles:
        return out_smiles
    else:
        iteration -= 1
        if iteration > 0:
            out_smiles = randomize_smiles(smiles,random_smiles,iteration)
            return out_smiles
        return (False)

    
# Takes single smiles and augment it based on count (Tries to get new smiles out of the smile is provided in iteration)
def augment_smiles(count,iteration,smiles):
    random_smiles = []
    for i in range(count):
        if smiles != None:
            out_smiles = randomize_smiles(smiles,random_smiles,iteration=iteration)
            if out_smiles:
                random_smiles.append(out_smiles)
            else:
                break
        
    return random_smiles


# To store the augmented smiles in a file
def unpack_and_write_list(smiles,label=None,filename=None):
    if filename == None:
        print ("Filename not provided")
        return None
        
    for entry in smiles:
        if type(entry) == list:
            unpack_and_write_list(entry,label,filename)
        else:
            if label == None:
                filename.write(entry + "\n")
            else:
                filename.write(entry + "," + str(label) + "\n")

                
# To augment a list of smiles or df of smiles with header ("Smiles,Label")
# N_rounds = number of times for augmentation (List of number based on per label augmentation or a number)
# iteration = Number of trials to get a new smiles
# data_set_type = To determine the output filename
# Number_of_workers = to run in pool of threads
def smiles_augmentation(df, N_rounds=1,iteration=5,data_set_type="train",Number_of_workers=1,CV=False):
    
    try:
        os.mkdir("data")
        os.mkdir("data/classification")
    except:
        pass
    
    if CV:
        #############
        # In case of CV do this
        folder_name = data_set_type.split(".")[0]
        try:
            os.mkdir("data/classification/" + str(folder_name))
        except:
            pass
        filename = "data/classification/" + str(folder_name) + "/"+ str(data_set_type.split(".")[1]) + "_aug_canonical_smiles.csv"
        ###################
    else:
        #########
        # In case of normal do this
        filename = "data/classification/" + str(data_set_type) + "_aug_canonical_smiles.csv"
        ##########
    
    aug_out = open(filename,"w")

    
    if type(df) == list:
        if type(N_rounds) == list:
            print ("N_rounds got a list not a number")
            return None
        
        if CV:
            augmented_smiles = augment_smiles(N_rounds, iteration,df[0])
        else:
            p = Pool(Number_of_workers)
            func = partial(augment_smiles, N_rounds, iteration)
            augmented_smiles = list(tqdm.tqdm(p.imap(func, df), total=len(df),leave=False))
            p.close()
        
        unpack_and_write_list(augmented_smiles,filename=aug_out)
        
        aug_out.close()
        
        return (open(filename,"r").read().split())
    
    else:
        aug_out.write("Smiles,Label\n")
        
        labels = []
        for label in df.groupby('Label'):
            labels.append(label[0])

        augmentation_list = []
        if type(N_rounds) == list:
            assert(len(N_rounds) == len(labels))
            augmentation_list = list(map(int, N_rounds))
        else:
            for i in range(len(labels)):
                augmentation_list.append(N_rounds)
                
        p = Pool(Number_of_workers)
        for label,augmentation in zip(labels,augmentation_list):

            canonical_smiles = df[df['Label'] == label]['Smiles'].to_list()

            
            func = partial(augment_smiles, augmentation, iteration)
            augmented_smiles = list(tqdm.tqdm(p.imap(func, canonical_smiles), total=len(canonical_smiles),leave=False))
            

            #print ("Saving data for label = " + str(label))

            unpack_and_write_list(augmented_smiles,label,filename=aug_out)

            unpack_and_write_list(canonical_smiles,label,filename=aug_out)
            
            #print ("Saved data for label = " + str(label))
        p.close()
        
        aug_out.close()

        return (pd.read_csv(filename, header=0).sample(frac=1).reset_index(drop=True))
    
# Get label wise augmentation needed to balance the data
# Input is dataframe with header ("Smiles,Label")
def get_augmentation_list(df,number_of_augmentation=1):
    label_count_df = df.groupby('Label').count()
    label_count_list = []
    for entry in range(len(label_count_df)):
        label_count_list.append(label_count_df.iloc[entry][0])

    augmentation_list = []
    max_value = max(label_count_list)
    for entry in label_count_list:
        augmentation_list.append((max_value/entry))
    
    augmentation_list = [entry*number_of_augmentation for entry in augmentation_list]
    
    return (augmentation_list)


# Given a CID, list of CIDs, dict (key as cid) of CIDs --> Gives out the smiles for the CID
def fetch_smiles(processed_query,type_smiles="isomeric",get_from="SDF",out_sub_name=0,folder_name="",remove_sdf=False):
    
    # TO GET SMILES FROM SDF FORMAT
    if get_from == "SDF":
        URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + processed_query + "/SDF"
        r = requests.get(URL)
        
        if folder_name != "":
            if not os.path.exists("structure_files/" + str(folder_name)):
                os.system("mkdir structure_files/" + str(folder_name))
            output_filename = "structure_files/" + str(folder_name) + "/file_" + str(out_sub_name) + ".sdf"
        else:
            output_filename = "structure_files/file_" + str(out_sub_name) + ".sdf"
            
        with open(output_filename, 'wb') as f:
            f.write(r.content)
            
        sppl = Chem.SDMolSupplier(output_filename)
        cid_list = []
        with open(output_filename, 'r') as f:
            f_lines = f.readlines()
            next_line = False
            for i,line in enumerate(f_lines):
                if i == 0:
                    try:
                        cid_list.append(int(line.split()[0]))
                    except:
                        error_fetching_file = open(output_filename[:-4] + ".txt","w")
                        error_fetching_file.write(processed_query)
                        error_fetching_file.close()
                        if remove_sdf:
                            os.system("rm  " + output_filename)
                        return None
                if next_line:
                    cid_list.append(int(line.split()[0]))
                    next_line = False
                    
                if "$$$$" in line:
                    next_line = True
                
        cid_smiles = {}
        for i,mol in enumerate(sppl):
            if mol is not None:# some compounds cannot be loaded.
                cid_smiles[cid_list[i]] = Chem.MolToSmiles(mol)
        
        if remove_sdf:
            os.system("rm  " + output_filename)
            
        return cid_smiles
    
    # TO GET SMILES FROM CANONICAL OR ISOMERIC ENTRY
    else:    
        if type_smiles == "canonical":
            URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + processed_query + "/property/CanonicalSMILES/json"
        else:
            URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/" + processed_query + "/property/IsomericSMILES/json"

        r = requests.get(URL) 
        cid_smiles = {}
        try:
            for entry in r.json()['PropertyTable']['Properties']:
                if type_smiles == "canonical":
                    cid_smiles[entry["CID"]] = entry["CanonicalSMILES"]
                else:
                    cid_smiles[entry["CID"]] = entry["IsomericSMILES"]
        except:
            return None
        return (cid_smiles)

def get_smiles_from_cid(query,type_smiles="isomeric",get_from="SDF",folder_file_name="",save_output=False,remove_sdf=False):
    assert(type(query) == int or type(query) == list or type(query) == dict)
    
    if get_from=="SDF":
        if not os.path.exists("structure_files"):
            os.system("mkdir structure_files")
        
    if type(query) == list or type(query) == dict:
        output_dict = {}
        processed_query = ""
        loop = tqdm.tqdm(enumerate(query), total=len(query),leave=False)
        block_count = 0
        for i,cid in loop:
            try:
                cid = int(cid)
            except:
                return ("Error in CID = " + str(cid))
            processed_query += str(cid) + ","
            
            if (i + 1) % 500 == 0 or (i + 1) == len(query):
                processed_query = processed_query[:-1]
                fetched_dict = fetch_smiles(processed_query,
                                            type_smiles=type_smiles,
                                            get_from=get_from,
                                            out_sub_name=block_count,
                                            folder_name=folder_file_name,
                                            remove_sdf=remove_sdf)
                if fetched_dict != None:
                    output_dict.update(fetched_dict)
                    if save_output:
                        current_batch_filename = "structure_files/" + str(folder_file_name) + "/file_" + str(block_count) + "_fetched.txt"
                        current_batch_file = open(current_batch_filename ,"w")
                        for cid in fetched_dict:
                            current_batch_file.write(str(cid) + " " + fetched_dict[cid] + "\n")
                        current_batch_file.close()
                    
                time.sleep(0.20)
                processed_query = ""
                block_count += 1
        
        if save_output:
            output_filename = open("structure_files/" + str(folder_file_name) + "_final_file.txt","w")
            for cid in output_dict:
                output_filename.write(str(cid) + " " + output_dict[cid] + "\n")
            output_filename.close()
            return None
        else:
            return output_dict

    if type(query) == int:
        processed_query = str(query)
        output_dict = fetch_smiles(processed_query)
        
        if save_output:
            output_filename = open("structure_files/" + str(folder_file_name) + ".txt","w")
            for cid in output_dict:
                output_filename.write(str(cid) + " " + output_dict[cid] + "\n")
            output_filename.close()
            return output_dict
        else:
            return output_dict
    

# Split data with label, splits to train, valid, and test
# df = dataframe of Smiles,Label
# train_percentage = train percentage of data
# valid_percentage = valid percentage of data
# test_percentage will be [1 - (train_percentage + valid_percentage)]
# returns dataframe of Smiles,Label
def list_to_dict_with_label(input_list,label):
    return {entry:label for entry in input_list}

def dict_to_label(input_dict):
    return pd.DataFrame(input_dict.items(),columns=["Smiles", "Label"])

def split_data_with_label(df,train_percentage=0.7,valid_percentage=None):
    if type(df) == dict:
        df = dict_to_label(df)
        
    if valid_percentage == None:
        valid_percentage = (1-train_percentage)/2
        test_percentage = (1-train_percentage)/2
    else:
        test_percentage = 1 - (train_percentage + valid_percentage)
    
    labels = []
    for label in df.groupby('Label'):
        labels.append(label[0])
    
    train_dict = {}
    valid_dict = {}
    test_dict = {}
    for i,label in enumerate(labels):
        canonical_smiles = np.array(df[df['Label'] == label]['Smiles'].to_list())
        
        truth_array = np.random.rand(len(canonical_smiles))
        
        train_data = canonical_smiles[truth_array < train_percentage]
        valid_data = canonical_smiles[np.logical_and(train_percentage < truth_array,truth_array < (valid_percentage + train_percentage))]
        test_data = canonical_smiles[(train_percentage + valid_percentage) < truth_array]
        
        train_dict.update(list_to_dict_with_label(train_data,label))
        valid_dict.update(list_to_dict_with_label(valid_data,label))
        test_dict.update(list_to_dict_with_label(test_data,label))
            
    train_df = pd.DataFrame(columns=["Smiles","Label"]) 
    train_df["Smiles"] = list(train_dict.keys())
    train_df["Label"]  = list(train_dict.values())
    
    valid_df = pd.DataFrame(columns=["Smiles","Label"]) 
    valid_df["Smiles"] = list(valid_dict.keys())
    valid_df["Label"]  = list(valid_dict.values())
    
    test_df = pd.DataFrame(columns=["Smiles","Label"]) 
    test_df["Smiles"] = list(test_dict.keys())
    test_df["Label"]  = list(test_dict.values())
    
    return (train_df.sample(frac=1),valid_df.sample(frac=1),test_df.sample(frac=1))



# Split data without label, splits to train, valid, and test
# input_list = input list of smiles to be split
# train_percentage = train percentage of data
# valid_percentage = valid percentage of data
# test_percentage will be [1 - (train_percentage + valid_percentage)]
# returns list of smiles with split
def split_data_without_label(input_list,train_percentage=0.7,valid_percentage=None):
    if valid_percentage == None:
        valid_percentage = (1-train_percentage)/2
        test_percentage = (1-train_percentage)/2
    else:
        test_percentage = 1 - (train_percentage + valid_percentage)
        
    canonical_smiles = np.array(input_list)
    
    truth_array = np.random.rand(len(canonical_smiles))
    
    train_data = list(canonical_smiles[truth_array < train_percentage])
    valid_data = list(canonical_smiles[np.logical_and(train_percentage < truth_array,truth_array < (valid_percentage + train_percentage))])
    test_data = list(canonical_smiles[(train_percentage + valid_percentage) < truth_array])
    
    return (train_data,valid_data,test_data)