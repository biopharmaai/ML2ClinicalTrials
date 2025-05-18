'''

(I). Trial_Dataset for prediction
(II). Trial_Dataset_Complete for interpretation
(III). SMILES lst 
(IV). disease lst icd-code 

'''
import pdb
import torch, csv, os
import pandas as pd
import numpy as np
from torch.utils import data 
from torch.utils.data.dataloader import default_collate
from trialbench.models.molecule_encode import smiles2mpnnfeature
from trialbench.models.protocol_encode import protocol2feature, load_sentence_2_vec
import json
from importlib import resources

sentence2vec = load_sentence_2_vec() 

def read_from_local(target, phase):
    # with resources.path('trialbench', 'data') as data_path:
    #     DIR_PATH = str(data_path)
    
    phase = 'All' if phase is None else phase
    phase = phase.capitalize() if phase.lower() != 'all' else 'All'
    if phase in ['1', '2', '3', '4']:
        phase = 'Phase' + phase
    elif phase == 'All':
        pass
    else:
        raise ValueError(f"Invalid phase: {phase}. Phase should be '1', '2', '3', '4' or 'All'.")
    X_train, y_train, X_test, y_test = [], [], [], []

    # Load dataset from local file
    dataset_path = os.path.join('data/', target, phase)
    for data_name in os.listdir(dataset_path):
        if 'train_x' in data_name:
            X_train = pd.read_csv(os.path.join(dataset_path, data_name))
            X_train = X_train.replace('nan', np.nan)
        elif 'train_y' in data_name:
            if target not in ['dose_cls']:
                y_train = pd.read_csv(os.path.join(dataset_path, data_name))
            else:
                y_train = pd.read_csv(os.path.join(dataset_path, data_name))['Avg']
            y_train = y_train.replace('nan', np.nan)
        elif 'test_x' in data_name:
            X_test = pd.read_csv(os.path.join(dataset_path, data_name))
            X_test = X_test.replace('nan', np.nan)
        elif 'test_y' in data_name:
            if target not in ['dose_cls']:
                y_test = pd.read_csv(os.path.join(dataset_path, data_name))
            else:
                y_test = pd.read_csv(os.path.join(dataset_path, data_name))['Avg']
            y_test = y_test.replace('nan', np.nan)

    if phase == 'All':
        if isinstance(X_train, list) and len(X_train) > 0:
            X_train = pd.concat(X_train, axis=0)
        if isinstance(y_train, list) and len(y_train) > 0:
            y_train = pd.concat(y_train, axis=0)
        if isinstance(X_test, list) and len(X_test) > 0:
            X_test = pd.concat(X_test, axis=0)
        if isinstance(y_test, list) and len(y_test) > 0:
            y_test = pd.concat(y_test, axis=0)

    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)
    return X_train, y_train, X_test, y_test


class ADMET_Dataset(data.Dataset):
    def __init__(self, smiles_lst, label_lst):
        self.smiles_lst = smiles_lst 
        self.label_lst = label_lst 
    
    def __len__(self):
        return len(self.smiles_lst)

    def __getitem__(self, index):
        return self.smiles_lst[index], self.label_lst[index]

def admet_collate_fn(x):
    smiles_lst = [i[0] for i in x]
    label_vec = default_collate([int(i[1]) for i in x])  ### shape n, 
    return [smiles_lst, label_vec]


def smiles_txt_to_lst(text):
    """
        "['CN[C@H]1CC[C@@H](C2=CC(Cl)=C(Cl)C=C2)C2=CC=CC=C12', 'CNCCC=C1C2=CC=CC=C2CCC2=CC=CC=C12']" 
    """
    text = text[1:-1]
    lst = [i.strip()[1:-1] for i in text.split(',')]
    return lst 

def icdcode_text_2_lst_of_lst(text):
    text = text[2:-2]
    lst_lst = []
    for i in text.split('", "'):
        i = i[1:-1]
        lst_lst.append([j.strip()[1:-1] for j in i.split(',')])
    return lst_lst 



def csv_ours_feature_2_dataloader(csvfile, shuffle, batch_size, phase='Phase 1', label='outcome'):
    # with open(csvfile, 'r') as csvfile:
    # 	rows = list(csv.reader(csvfile, delimiter=','))[1:]
    ## nctid,status,why_stop,label,phase,diseases,icdcodes,drugs,smiless,criteria
    X_train = pd.read_csv(csvfile)
    X_train = X_train[X_train['phase'] == phase]
    labels = pd.read_csv(csvfile.replace('_x.csv', '_y.csv'))[label]
    def mapper(x):
        if x == 'poor enrollment':
            return 0
        elif x == 'efficacy':
            return 1
        elif x == 'safety':
            return 2
        elif x == 'Others':
            return 3
    labels = labels[X_train.index].apply(mapper)
    nctid_lst = X_train['nctid'].tolist()
    label_lst = labels.tolist()
    icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    drugs_lst = X_train['intervention/intervention_name'].tolist()
    smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    criteria_lst = X_train['eligibility/criteria/textblock'].tolist()
    dataset = Trial_Dataset(nctid_lst, label_lst, smiles_lst, icdcode_lst, criteria_lst)
    data_loader = data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = trial_collate_fn)
    return data_loader


def csv_three_feature_2_complete_dataloader(csvfile, shuffle, batch_size):
    with open(csvfile, 'r') as csvfile:
        rows = list(csv.reader(csvfile, delimiter=','))[1:]	
    nctid_lst = [row[0] for row in rows]
    status_lst = [row[1] for row in rows]
    why_stop_lst = [row[2] for row in rows]
    label_lst = [row[3] for row in rows]
    phase_lst = [row[4] for row in rows]
    diseases_lst = [row[5] for row in rows]
    icdcode_lst = [row[6] for row in rows]
    drugs_lst = [row[7] for row in rows]
    smiles_lst = [row[8] for row in rows]
    new_drugs_lst, new_smiles_lst = [], []
    criteria_lst = [row[9] for row in rows] 
    dataset = Trial_Dataset_Complete(nctid_lst, status_lst, why_stop_lst, label_lst, phase_lst, 
                                        diseases_lst, icdcode_lst, drugs_lst, smiles_lst, criteria_lst)
    data_loader = data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, collate_fn = trial_complete_collate_fn)
    return data_loader 

def smiles_txt_to_2lst(smiles_txt_file):
    with open(smiles_txt_file, 'r') as fin:
        lines = fin.readlines() 
    smiles_lst = [line.split()[0] for line in lines]
    label_lst = [int(line.split()[1]) for line in lines]
    return smiles_lst, label_lst 

def generate_admet_dataloader_lst(batch_size):
    datafolder = "data/ADMET/cooked/"
    name_lst = ["absorption", 'distribution', 'metabolism', 'excretion', 'toxicity']
    dataloader_lst = []
    for i,name in enumerate(name_lst):
        train_file = os.path.join(datafolder, name + '_train.txt')
        test_file = os.path.join(datafolder, name +'_valid.txt')
        train_smiles_lst, train_label_lst = smiles_txt_to_2lst(train_file)
        test_smiles_lst, test_label_lst = smiles_txt_to_2lst(test_file)
        train_dataset = ADMET_Dataset(smiles_lst = train_smiles_lst, label_lst = train_label_lst)
        test_dataset = ADMET_Dataset(smiles_lst = test_smiles_lst, label_lst = test_label_lst)
        train_dataloader = data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
        test_dataloader = data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
        dataloader_lst.append((train_dataloader, test_dataloader))
    return dataloader_lst 























