import tqdm
import numpy as np

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

# MolTokenizer_spe_sos_eos for initilizing spe tokenizing with sos and eos tokens
# call by MolTokenizer_spe_sos_eos(token_path)
# token path is the path to tokens created while training SPE
class MolTokenizer_spe_sos_eos():
    def __init__(self, lang = 'en',token_path=""):
        self.lang = lang
        spe_vob= codecs.open(token_path)
        self.spe = SPE_Tokenizer(spe_vob)
        
    def tokenizer(self, smiles):
        smiles = "[BOS]" + smiles
        tokens = self.spe.tokenize(smiles).split()
        return tokens
        
    def add_special_cases(self, toks):
        pass
    
# MolTokenizer_atomwise_sos_eos for initilizing spe tokenizing with sos and eos tokens
# call by MolTokenizer_atomwise_sos_eos(token_path)
# token path is the path to tokens created while training SPE
class MolTokenizer_atomwise_sos_eos():
    def __init__(self, lang = 'en',token_path=""):
        self.lang = lang
        
    def tokenizer(self, smiles):
        smiles = "[BOS]" + smiles
        tokens = atomwise_tokenizer(smiles)
        return tokens
        
    def add_special_cases(self, toks):
        pass 