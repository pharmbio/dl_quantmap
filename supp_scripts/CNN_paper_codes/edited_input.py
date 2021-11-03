#!/usr/bin/env python 
# coding:utf-8

import time, argparse, gc, os

import numpy as np
import pandas as pd

from rdkit import Chem

from feature import *
import SCFPfunctions as Mf
#import SCFPmodel as Mm

# chainer v2
'''import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Reporter, report, report_scope
from chainer import Link, Chain, ChainList, training
from chainer.datasets import tuple_dataset
from chainer.training import extensions'''


xp=np
#xp=cp

print('Making Training  Dataset...')
file='NR-AR-LBD_wholetraining.smiles.txt'
print('Loading smiles: ', file)
smi = Chem.SmilesMolSupplier(file,delimiter=' ',titleLine=False)
mols = [mol for mol in smi if mol is not None]

F_list, T_list = [],[]
for mol in mols:
    if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > 400: print("too long mol was ignored")
    else:
        F_list.append(mol_to_feature(mol,-1,400))
        T_list.append(mol.GetProp('_Name'))
Mf.random_list(F_list)
Mf.random_list(T_list)

data_t = xp.asarray(T_list, dtype=xp.int32).reshape(-1,1)
data_f = xp.asarray(F_list, dtype=xp.float32).reshape(-1,1,400,lensize)
print(data_t.shape, data_f.shape)
train_dataset = (data_f,data_t)
print (train_dataset)
#train_dataset = datasets.TupleDataset(data_f, data_t) 
