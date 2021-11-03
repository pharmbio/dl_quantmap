import tqdm
import numpy as np
import functools

from multiprocessing import Pool
from functools import partial

import codecs
from SmilesPE.pretokenizer import atomwise_tokenizer
from SmilesPE.pretokenizer import kmer_tokenizer
from SmilesPE.learner import *
from SmilesPE.tokenizer import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence

from torchtext import data


# train spe-tokenizer
# smiles = list of smiles
# token_path = output path
# min_frequency = minimum pair frequency to consider
# augmentation = number to augmentation for each smiles entry
def train_spe_tokenizer(smiles,token_path,min_frequency=200,augmentation=0):
    assert(type(smiles) == list)
    output = codecs.open(token_path, 'w')
    learn_SPE(smiles, output, 30000, min_frequency, augmentation, verbose=False, total_symbols=True)
    
# given the model,criterion,dataloader,device returns loss,accuracy,prediction_list
# prediction_list = list of [[ground_truth],[predictions]]
def test(model,criterion,val_dl,device="cpu",get_prediction_list=True):
    model.eval()
    total_loss = []
    accuracy = []
    
    real_and_predictions = [[],[]]
    
    with torch.no_grad():        
        val_dl.create_batches()
        for i,batch in enumerate(val_dl.batches):
            batch_text = [example["text"] for example in batch]
            batch_label = torch.tensor([example["label"] for example in batch])
            
            x_padded = pad_sequence(batch_text,batch_first=False, padding_value=0)
            xvalc = x_padded.to(device)
            yvalc = batch_label.to(device)
            
            # Forward prop
            output_val = model(xvalc.long(),yvalc)
            #print (output_val)
            accuracy.append(get_accuracy(output_val,yvalc))
            loss_val = criterion(output_val, yvalc)
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
        val_dl.create_batches()
        for i,batch in enumerate(val_dl.batches):
            batch_text = [example["text"] for example in batch]
            batch_label = torch.tensor([example["label"] for example in batch])
            
            x_padded = pad_sequence(batch_text,batch_first=False, padding_value=0)
            xvalc = x_padded.to(device)
            yvalc = batch_label.to(device)
            
            # Forward prop
            output_val = model(xvalc.long(),yvalc)
            accuracy.append(get_accuracy(output_val,yvalc))
            loss_val = criterion(output_val, yvalc)
            total_loss.append(loss_val.item())

    return (sum(total_loss)/(i+1),sum(accuracy)/(len(accuracy)))


# pretrain validate the model
# find loss and accuracy for train set
# model = model as input
# criterion = criterion as input
# optimizer = optimizer as input
# train_dl = train dataloader
# device = device for calculation
def pretrain_validate(model,criterion,valid_dl,device="cuda",padding_value=0):
    model.eval()
    total_loss = []
    accuracy = []
    
    with torch.no_grad():
        valid_dl.create_batches()
        for i,batch in enumerate(valid_dl.batches):
            batch_text = [example["text"] for example in batch]
            
            x_padded = pad_sequence(batch_text,batch_first=False, padding_value=padding_value) 
            xbc = x_padded.to(device)

            # Forward prop
            output = model(xbc.long(),xbc.long())

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)

            ybc = xbc[1:].view(-1)

            accuracy.append(get_accuracy(output,ybc))
            loss_train = criterion(output,ybc)
            total_loss.append(loss_train.item())

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
    
    train_dl.create_batches()
    
    for i,batch in enumerate(train_dl.batches):
        batch_text = [example["text"] for example in batch]
        batch_label = torch.tensor([example["label"] for example in batch])
        
        x_padded = pad_sequence(batch_text,batch_first=False, padding_value=0)        
        xbc = x_padded.to(device)
        ybc = batch_label.to(device)
        
        optimizer.zero_grad()
        # Forward prop
        output_train = model(xbc.long(),ybc)
        accuracy.append(get_accuracy(output_train,ybc))
        loss_train = criterion(output_train, ybc)
        
        # Back prop
        loss_train.backward()        
        optimizer.step()
        total_loss.append(loss_train.item())

    return (sum(total_loss)/(i+1),sum(accuracy)/(len(accuracy)))


# pre train the model
# find loss and accuracy for train set
# model = model as input
# criterion = criterion as input
# optimizer = optimizer as input
# train_dl = train dataloader
# device = device for calculation
def pretrain_train(model,criterion,optimizer,train_dl,device="cuda",padding_value=0,clip=1):
    model.train()
    total_loss = []
    accuracy = []
    
    train_dl.create_batches()
    
    for i,batch in enumerate(train_dl.batches):
        batch_text = [example["text"] for example in batch]

        x_padded = pad_sequence(batch_text,batch_first=False, padding_value=padding_value)     
        xbc = x_padded.to(device)

        optimizer.zero_grad()
        output_train = model(xbc.long(),xbc.long())
        
        output_dim = output_train.shape[-1]
        output = output_train[1:].view(-1, output_dim)

        ybc = xbc[1:].view(-1)

        accuracy.append(get_accuracy(output,ybc))
        
        loss_train = criterion(output,ybc)
        loss_train.backward() 
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        total_loss.append(loss_train.item())
        
    return (sum(total_loss)/(i+1),sum(accuracy)/(len(accuracy)))


# Get accuracy for data given yhat and y
# yhat = predictions
# y = ground truth
def get_accuracy(yhat,y):
    softmax = torch.exp(yhat.float())
    prob = softmax.cpu().detach().numpy()
    predictions = np.argmax(prob, axis=1)
    y_truth = y.cpu().detach().numpy()
    accuracy_check = (y_truth==predictions)
    count = np.count_nonzero(accuracy_check)
    accuracy = (count/len(accuracy_check))
    return accuracy


# receives indexed x and y and returns bucket iterator
# x_indexed = indexed x (tokenized and indexed)
# y = list of labels
# batch_size = batch size
# device = "cpu" or "cuda"
# returns bucket iterator of specified size
def make_bucket_iterator(x_indexed,y=None,batch_size=32,device="cpu"):
    list_of_dict = []
    for i,entry in enumerate(x_indexed):
        if y != None:
            list_of_dict.append(dict(text=entry,label=int(y[i])))
        else:
            list_of_dict.append(dict(text=entry))
            
    bucket_iterator = data.BucketIterator(
                        list_of_dict,
                        sort = False,
                        sort_within_batch=True,
                        sort_key=lambda x: len(x['text']),
                        batch_size = batch_size,
                        device = device)        
    return (bucket_iterator)


# Takes list (single molecule) tokens for a molecule and returns indexed tokens
# molecule = list of tokens
# word_index = {word:index} dict
# returns indexed tokens for the molecule
def convert_token_to_index(molecule,word_index):
    idxs = []
    for token in molecule:
        if token in word_index:
            idxs.append(word_index[token])
        else:
            idxs.append(word_index["<UNK>"])
    return (torch.tensor(idxs))
    
    
# Takes list of list of tokens of molecules and returns indexed tokens
# molecule_token_list = list of list of tokens
# word_index = {word:index} dict
# returns list of list indexed tokens for the molecules
def convert_token_to_index_multi(molecule_token_list,word_index):
    indexed_tokens = []
    loop = tqdm.tqdm(molecule_token_list, total=len(molecule_token_list),leave=False)
    for molecule in loop:
        indexed_tokens.append(convert_token_to_index(molecule,word_index))
    return (indexed_tokens)


# MolTokenizer_spe for initilizing spe tokenizing
# call by MolTokenizer_spe(token_path)
# token path is the path to tokens created while training SPE
class MolTokenizer_spe():
    def __init__(self,token_path, lang = 'en'):
        self.lang = lang
        spe_vob= codecs.open(token_path)
        self.spe = SPE_Tokenizer(spe_vob)
        
    def tokenizer(self, output_label=None,smiles=None):
        tokens = self.spe.tokenize(smiles).split()
        return tokens
        
    def add_special_cases(self, toks):
        pass

    
    
# MolTokenizer_spe_sos_eos for initilizing spe tokenizing with sos and eos tokens
# call by MolTokenizer_spe_sos_eos(token_path)
# token path is the path to tokens created while training SPE
class MolTokenizer_spe_sos_eos():
    def __init__(self,token_path, lang = 'en'):
        self.lang = lang
        spe_vob= codecs.open(token_path)
        self.spe = SPE_Tokenizer(spe_vob)
        
    def tokenizer(self, output_label=None,smiles=None):
        tokens = self.spe.tokenize(smiles).split()
        tokens.insert(0, "<SOS>") 
        tokens.append("<EOS>") 
        return tokens
        
    def add_special_cases(self, toks):
        pass
    
# Unpack list of list of vocab created by make_vocabulary
# for the set of smiles
# vocab_list = input list of vocab by make_vocabulary
# vocab_unpacked = initilize by blank list
def unpack_vocab_list(vocab_list,vocab_unpacked):
    for entry in vocab_list:
        if type(entry) == list:
            unpack_vocab_list(entry,vocab_unpacked)
        else:
            vocab_unpacked.append(entry)
            
    return (vocab_unpacked)


# initilizes MolTokenizer_spe/MolTokenizer_spe_sos_eos and creates tokens for the given list of molecules
# input_list = list of smiles for tokenization
# token_path = path to tokens created while training SPE
# Number_of_workers = number of threads to use
# sos_eos_tokens = set to True to include sos and eos tokens
# Returns unpacked list of vocabs
def make_vocabulary(input_list,token_path="",Number_of_workers=1,sos_eos_tokens=False,tokenization="atomwise"):
    if tokenization == "atomwise":
        if sos_eos_tokens:
            tok = MolTokenizer_atomwise_sos_eos()
        else:
            tok = MolTokenizer_atomwise()
    else:
        if sos_eos_tokens:
            tok = MolTokenizer_spe_sos_eos(token_path)
        else:
            tok = MolTokenizer_spe(token_path)
        
    p = Pool(Number_of_workers)
    func = partial(tok.tokenizer, None)
    vocab_list = list(tqdm.tqdm(p.imap(func, input_list), total=len(input_list),leave=False))
    p.close()
    vocab_unpacked = []
    return (list(set(unpack_vocab_list(vocab_list,vocab_unpacked))))


# make dict of word_index and index_word
# for a given input vocab from unpack_vocab_list
# vocab = input vocab from unpack_vocab_list
# returns word_index and index_word dicts
def make_word_index(vocab):
    word_index = {}
    index_word = {}
    for i,entry in enumerate(vocab):
        word_index[entry] = i
        index_word[i] = entry
    return (word_index,index_word)


# read existing vocab file
# vocab_input_filename = input path for vocab file created by create_vocab_file
# returns word_index and index_word dicts
def read_vocab_file(vocab_input_filename):
    vocab = open(vocab_input_filename,"r").read().strip("[]").replace("'", "").replace(" ", "").split(",")
    word_index,index_word = make_word_index(vocab)
    return (word_index,index_word)


# create vocab from input dataframe
# input_df = input dataframe of ["Smiles","Label"]
# token_path = path to tokens created while training SPE
# output_path = location to save the file
# Number_of_workers = number of threads to use
# sos_eos_tokens = set to True to include sos and eos tokens
# returns word_index and index_word dicts 
def create_vocab_file_spe(input_df,token_path,Number_of_workers,output_path,sos_eos_tokens=False):
    smiles_list = input_df["Smiles"].to_list()
    
    vocab = ["<UNK>"]
    
    vocab.extend(make_vocabulary(smiles_list,token_path,Number_of_workers,sos_eos_tokens,tokenization="SPE"))
    
    vocab.sort()
    
    vocab.insert(0,"<PAD>")
    
    word_index,index_word = make_word_index(vocab)

    vocab_output = open(output_path,"w")
    vocab_output.write(str(vocab))
    vocab_output.close()
    
    return (word_index,index_word)
    

# for a input dataframe, converts to tokens using MolTokenizer using token_path provided
# input_df = input dataframe of ["Smiles","Label"]
# upper_cutoff = maximum sequence length to be considered
# lower_cutoff = minimum sequence length to be considered
# token_path = path to tokens created while training SPE
# sos_eos_tokens = set to True to include sos and eos tokens
def convert_smiles_to_tokens(input_df,lower_cutoff=2,upper_cutoff=150,Number_of_workers=1,token_path="",sos_eos_tokens=False,tokenization="atomwise"):
    
    if tokenization == "atomwise":
        if sos_eos_tokens:
            tok = MolTokenizer_atomwise_sos_eos()
        else:
            tok = MolTokenizer_atomwise()
    else:
        if sos_eos_tokens:
            tok = MolTokenizer_spe_sos_eos(token_path)
        else:
            tok = MolTokenizer_spe(token_path)
    
    if type(input_df) == list:
        x = []
        y = []
        p = Pool(Number_of_workers)
        smiles_list = input_df
        
        func = partial(tok.tokenizer, None)
        tokens = list(tqdm.tqdm(p.imap(func, smiles_list), total=len(smiles_list),leave=False))

        for entry in tokens:
            if  lower_cutoff < len(entry) <= upper_cutoff:
                x.append(entry)
        p.close()
        
        return (x)
        
    else:
        labels = []
        for label in input_df.groupby('Label'):
            labels.append(label[0])

        x = []
        y = []
        p = Pool(Number_of_workers)
        for label in labels:
            smiles_list = input_df[input_df['Label'] == label]['Smiles'].to_list()


            func = partial(tok.tokenizer, None)
            tokens = list(tqdm.tqdm(p.imap(func, smiles_list), total=len(smiles_list),leave=False))

            for entry in tokens:
                if  lower_cutoff < len(entry) <= upper_cutoff:
                    x.append(entry)
                    y.append(label)
        p.close()
        
        return (x,y)


# MolTokenizer_atomwise_sos_eos for initilizing spe tokenizing with sos and eos tokens
# call by MolTokenizer_atomwise_sos_eos(token_path)
# token path is the path to tokens created while training SPE
class MolTokenizer_atomwise_sos_eos():
    def __init__(self, lang = 'en'):
        self.lang = lang
        
    def tokenizer(self, output_label=None,smiles=None):
        tokens = atomwise_tokenizer(smiles)
        tokens.insert(0, "<SOS>") 
        tokens.append("<EOS>") 
        return tokens
        
    def add_special_cases(self, toks):
        pass 

    
# MolTokenizer_atomwise for initilizing spe tokenizing
# call by MolTokenizer_atomwise(token_path)
# token path is the path to tokens created while training SPE
class MolTokenizer_atomwise():
    def __init__(self, lang = 'en'):
        self.lang = lang
        
    def tokenizer(self, output_label=None,smiles=None):
        tokens = atomwise_tokenizer(smiles)
        return tokens
        
    def add_special_cases(self, toks):
        pass        
    
# create vocab from input dataframe
# input_df = input dataframe of ["Smiles","Label"]
# token_path = path to tokens created while training SPE
# output_path = location to save the file
# Number_of_workers = number of threads to use
# sos_eos_tokens = set to True to include sos and eos tokens
# returns word_index and index_word dicts 
def create_vocab_file_atomwise(input_df,Number_of_workers,output_path,sos_eos_tokens=False):
    if type(input_df) == list:
        smiles_list = input_df
    else:
        smiles_list = input_df["Smiles"].to_list()
    
    vocab = ["<UNK>"]
    
    vocab.extend(make_vocabulary(smiles_list,Number_of_workers=Number_of_workers,sos_eos_tokens=sos_eos_tokens,tokenization="atomwise"))
    
    vocab.sort()
    
    vocab.insert(0,"<PAD>")
    
    word_index,index_word = make_word_index(vocab)

    vocab_output = open(output_path,"w")
    vocab_output.write(str(vocab))
    vocab_output.close()
    
    return (word_index,index_word)




def pretrain_convert_index_to_word(sentence,index_word_dict):
    sentence = list(sentence)
    output_list = [index_word_dict[word] for word in sentence]
    return (output_list)

def pretrain_get_real_and_predictions(y,yhat,index_word_dict={}):
    yhat = yhat.reshape(y.shape[0],y.shape[1],-1)
    softmax = torch.exp(yhat.float())
    prob = softmax.cpu().detach().numpy()
    predictions = np.argmax(prob, axis=2)
    
    input_sentences = []
    output_sentences = []
    
    for batch in y:
        input_sentences.append(pretrain_convert_index_to_word(batch.cpu().detach().numpy(),index_word_dict))
    input_sentences = (list(map(list, zip(*input_sentences))))
    
    for batch in predictions:
        output_sentences.append(pretrain_convert_index_to_word(batch,index_word_dict))
    output_sentences = (list(map(list, zip(*output_sentences))))
    
    return ([(functools.reduce(lambda a, b: str(a)+str(b), entry) ,functools.reduce(lambda a, b: str(a)+str(b),output_sentences[i])) for i,entry in enumerate(input_sentences)])


def pretrain_test(model,criterion,val_dl,device="cpu",padding_value=0,index_word_dict={},get_prediction_list=True):
    model.eval()
    total_loss = []
    accuracy = []
    
    real_and_predictions = []
    
    with torch.no_grad():        
        val_dl.create_batches()
        for i,batch in enumerate(val_dl.batches):
            batch_text = [example["text"] for example in batch]
            
            x_padded = pad_sequence(batch_text,batch_first=False, padding_value=padding_value)
            xbc = x_padded.to(device)
            
            # Forward prop
            output = model(xbc.long(),xbc.long())

            ybc = xbc[1:].view(-1) 
            
            if get_prediction_list and len(index_word_dict) > 0:
                real_and_predictions.append(pretrain_get_real_and_predictions(y=xbc,yhat=output,index_word_dict=index_word_dict))
                
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)

            accuracy.append(get_accuracy(output,ybc))
            loss_train = criterion(output,ybc)
            total_loss.append(loss_train.item())

    return (sum(total_loss)/(i+1),sum(accuracy)/(len(accuracy)),real_and_predictions)