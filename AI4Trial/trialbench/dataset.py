import torch, os, sys
torch.manual_seed(0) 

import socket
import pandas as pd
from torch.utils import data 
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from category_encoders import LeaveOneOutEncoder
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from trialbench.models.text_encode import text2feature
from trialbench.utils import smiles_txt_to_lst, protocol2feature, icdcode_text_2_lst_of_lst, sentence2vec, read_from_local
from trialbench.models.mesh_encode import mesh_term2feature

def get_dataloader(task, phase):
    '''
    Input:
        task: str, name of the task
        phase: str, phase of the clinical trial (e.g., "Phase 1", "Phase 2", "Phase 3", "Phase 4")
    Output: [train_loader, valid_loader, test_loader, num_classes, tabular_input_dim
        dataloaders:  tuple, containing train_loader, valid_loader, test_loader
        tabular_input_dim: int, number of tabular features
    '''
    if task == 'mortality_rate':
        return mortality_rate(phase)
    elif task == 'serious_adverse_rate':
        return serious_adverse_rate(phase)
    elif task == 'patient_dropout_rate':
        return patient_dropout_rate(phase)
    elif task == 'duration':
        return duration(phase)
    elif task == 'outcome':
        return outcome(phase)
    elif task == 'failure_reason':
        return failure_reason(phase) 
    elif task == 'serious_adverse_rate_yn':
        return serious_adverse_rate_yn(phase)
    elif task == 'patient_dropout_rate_yn':
        return patient_dropout_rate_yn(phase)
    elif task == 'mortality_rate_yn':
        return mortality_rate_yn(phase)
    elif task == 'dose':
        return dose(phase)
    elif task == 'dose_cls':
        return dose_cls(phase)

def data_loader_to_frame(dataloader, task):
    if 'dose' in task:
        all_data = {
            'nctid_lst': [],
            'label_lst': [],
            'smiles_lst': [],
            'mesh_lst': []
        }

        for nctid_lst, label_lst, smiles_lst, mesh_lst in dataloader:
            all_data['nctid_lst'].extend(nctid_lst)
            all_data['label_lst'].extend(label_lst)
            all_data['smiles_lst'].extend(smiles_lst)
            all_data['mesh_lst'].extend(mesh_lst)
    else:
        all_data = {
            'nctid_lst': [],
            'label_lst': [],
            'smiles_lst': [],
            'icdcode_lst': [],
            'criteria_lst': [],
            'tabular_lst': [],
            'text_lst': [],
            'mesh_lst': []
        }

        for nctid_lst, label_lst, icdcode_lst, smiles_lst, criteria_lst, tabular_lst, text_lst, mesh_lst in dataloader: 
            all_data['nctid_lst'].extend(nctid_lst)
            all_data['label_lst'].extend(label_lst)
            all_data['smiles_lst'].extend(smiles_lst)
            all_data['icdcode_lst'].extend(icdcode_lst)
            all_data['criteria_lst'].extend(criteria_lst)
            all_data['tabular_lst'].extend(tabular_lst)
            all_data['text_lst'].extend(text_lst)
            all_data['mesh_lst'].extend(mesh_lst)
    
    # print(f'all_data.keys(): {all_data.keys()}')

    return pd.DataFrame(all_data)

class Trial_Dataset_tabular(Dataset):
    def __init__(self, nctid_lst, label_lst, icdcode_lst, smiles_lst, criteria_lst, tabular_lst, text_lst, mesh_lst):
        self.nctid_lst = nctid_lst 
        self.label_lst = label_lst 
        self.smiles_lst = smiles_lst 
        self.icdcode_lst = icdcode_lst 
        self.criteria_lst = criteria_lst 
        self.tabular_lst = tabular_lst
        self.text_lst = text_lst
        self.mesh_lst = mesh_lst
    
    def __len__(self):
        return len(self.nctid_lst)

    def __getitem__(self, index):
        return self.nctid_lst[index], self.label_lst[index], self.smiles_lst[index], self.icdcode_lst[index], self.criteria_lst[index], self.tabular_lst[index], self.text_lst[index], self.mesh_lst[index]

def refine_year(x):
        # change to Months
        if type(x) == str:
            if 'Year' in x:
                number = eval(x.split(' ')[0])       
                return number * 12
            elif 'Month' in x:
                number = eval(x.split(' ')[0])
                return number
            elif 'Week' in x:
                number = eval(x.split(' ')[0])
                return number / 4.286
            elif 'Day' in x:
                number = eval(x.split(' ')[0])
                return number / 30
            elif 'Hour' in x:
                number = eval(x.split(' ')[0])
                return number / 30 / 24
            elif 'Minute' in x:
                number = eval(x.split(' ')[0])
                return number / 30 / 24 / 60
        return x

def trial_tabular_collate_fn(x):
    nctid_lst = [i[0] for i in x]     ### ['NCT00604461', ..., 'NCT00788957'] 
    label_vec = default_collate([i[1] for i in x])  ### shape n, 
    smiles_lst = [smiles_txt_to_lst(i[2]) for i in x]
    icdcode_lst = [icdcode_text_2_lst_of_lst(i[3]) for i in x]
    criteria_lst = [protocol2feature(i[4], sentence2vec) for i in x]
    tabular_lst = [i[5] for i in x]
    text_lst = [text2feature(i[6]) for i in x]
    mesh_lst = [mesh_term2feature(i[7]) for i in x]
    return [nctid_lst, label_vec, smiles_lst, icdcode_lst, criteria_lst, tabular_lst, text_lst, mesh_lst]

def trial_tabular_2_collate_fn(x):
    nctid_lst = [i[0] for i in x]     ### ['NCT00604461', ..., 'NCT00788957'] 
    label_vec = torch.tensor([i[1] for i in x])  ### shape n, 2
    smiles_lst = [smiles_txt_to_lst(i[2]) for i in x]
    icdcode_lst = [icdcode_text_2_lst_of_lst(i[3]) for i in x]
    criteria_lst = [protocol2feature(i[4], sentence2vec) for i in x]
    tabular_lst = [i[5] for i in x]
    text_lst = [text2feature(i[6]) for i in x]
    mesh_lst = [mesh_term2feature(i[7]) for i in x]
    return [nctid_lst, label_vec, smiles_lst, icdcode_lst, criteria_lst, tabular_lst, text_lst, mesh_lst]

def remove_unused_column(data):
    unused_list = []
    for col in data.columns:
        uni = len(data[col].unique())
        if uni <= 1:
            unused_list.append(col)
    data.drop(columns=unused_list, inplace=True)
    return data

def split_data(data, target, test_size):
    label = data[target]
    data = data.drop([target], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=test_size, random_state=123, shuffle=True)
    return X_train, y_train.values, X_test, y_test.values


def quantile_transform(X_train, X_valid, X_test):
    quantile_train = np.copy(X_train)
    qt = QuantileTransformer(random_state=55688, output_distribution='normal').fit(quantile_train)
    X_train = qt.transform(X_train)
    X_valid = qt.transform(X_valid)
    X_test = qt.transform(X_test)

    return X_train, X_valid, X_test

def unzip_file(zip_path, extract_to=None):
    import zipfile

    if extract_to is None:
        extract_to = os.path.dirname(zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        os.makedirs(extract_to, exist_ok=True)
        print(f"Unzipping files to: {extract_to}...")
        zip_ref.extractall(extract_to)
        
    print(f"Finish unzipping!")

def read_data(target, phase):
    from tqdm import tqdm
    import requests
    def download_data_to_local(url, save_path):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
    
        with open(save_path, 'wb') as file, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    size = file.write(chunk)
                    bar.update(size)
        
        print(f"The dataset is downloaded to: {save_path}")
    if not os.path.exists(f'data/{target}'):
        os.makedirs(f'data/{target}')
        try:
            url = f"https://zenodo.org/records/14975339/files/{target}.zip?download=1"
            save_path = f'data/{target}.zip'
            download_data_to_local(url, save_path)
            unzip_file(zip_path=f'data/{target}.zip',)
        except Exception as e:
            print(f"Error downloading data: {e}")
            exit(1)
    
    X_train, y_train, X_test, y_test = read_from_local(target, phase)
    
    return X_train, y_train, X_test, y_test
    

def mortality_rate(phase):
    # target = 'mortality-rate-prediction'
    target = 'mortality-event-prediction'
    X_train, y_train, X_test, y_test = read_data(target, phase)

    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # # maxmin scalar
    # min_val = y_train.min()
    # max_val = y_train.max()
    # y_train = (y_train - min_val) / (max_val - min_val)
    # y_valid = (y_valid - min_val) / (max_val - min_val)
    # y_test = (y_test - min_val) / (max_val - min_val)

    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train['Y/N'])
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    y_train = y_train['mortality_rate']
    y_valid = y_valid['mortality_rate']
    y_test = y_test['mortality_rate']
    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train.index.tolist()
    train_label_lst = y_train.values.tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_2_collate_fn)
    
    valid_nctid_lst = X_valid.index.tolist()
    valid_label_lst = y_valid.values.tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_2_collate_fn)

    test_nctid_lst = X_test.index.tolist()
    test_label_lst = y_test.values.tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_2_collate_fn)

    return train_loader, valid_loader, test_loader, 0, len(numerical_features + hot_features)
    

def mortality_rate_yn(phase):
    # target = 'mortality-rate-prediction'
    target = 'mortality-event-prediction'
    X_train, y_train, X_test, y_test = read_data(target, phase)
    y_train = y_train['Y/N']
    y_test = y_test['Y/N']

    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode' and c != 'eligibility/maximum_age']
    # print(drop_columns )
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train.index.tolist()
    train_label_lst = y_train.to_numpy().tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_collate_fn)
    
    valid_nctid_lst = X_valid.index.tolist()
    valid_label_lst = y_valid.to_numpy().tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    test_nctid_lst = X_test.index.tolist()
    test_label_lst = y_test.to_numpy().tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    return train_loader, valid_loader, test_loader, 1, len(numerical_features + hot_features)


def serious_adverse_rate(phase):
    # target = 'adverse-event-rate-prediction'
    target = 'serious-adverse-event-forecasting'
    X_train, y_train, X_test, y_test = read_data(target, phase)


    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)

    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # # maxmin scalar
    # min_val = y_train.min()
    # max_val = y_train.max()
    # y_train = (y_train - min_val) / (max_val - min_val)
    # y_valid = (y_valid - min_val) / (max_val - min_val)
    # y_test = (y_test - min_val) / (max_val - min_val)

    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train['Y/N'])
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]
    
    y_train = y_train['serious_adverse_rate']
    y_valid = y_valid['serious_adverse_rate']
    y_test = y_test['serious_adverse_rate']
    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train.index.tolist()
    train_label_lst = y_train.values.tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_2_collate_fn)
    
    valid_nctid_lst = X_valid.index.tolist()
    valid_label_lst = y_valid.values.tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_2_collate_fn)

    test_nctid_lst = X_test.index.tolist()
    test_label_lst = y_test.values.tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_2_collate_fn)

    return train_loader, valid_loader, test_loader, 0, len(numerical_features + hot_features)

def serious_adverse_rate_yn(phase):
    # target = 'adverse-event-rate-prediction'
    target = 'serious-adverse-event-forecasting'
    X_train, y_train, X_test, y_test = read_data(target, phase)
    y_train = y_train['Y/N']
    y_test = y_test['Y/N']
    
    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)

    if phase is not None:
        X_train = X_train[X_train['phase'] == phase]
        y_train = y_train[X_train.index]
        X_test = X_test[X_test['phase'] == phase]
        y_test = y_test[X_test.index]
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train.index.tolist()
    train_label_lst = y_train.to_numpy().tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_collate_fn)
    
    valid_nctid_lst = X_valid.index.tolist()
    valid_label_lst = y_valid.to_numpy().tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    test_nctid_lst = X_test.index.tolist()
    test_label_lst = y_test.to_numpy().tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    return train_loader, valid_loader, test_loader, 1, len(numerical_features + hot_features)

def patient_dropout_rate(phase):
    # target = 'patient_dropout_rate'
    target = 'patient-dropout-event-forecasting'
    X_train, y_train, X_test, y_test = read_data(target, phase)


    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)

    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # # maxmin scalar
    # min_val = y_train.min()
    # max_val = y_train.max()
    # y_train = (y_train - min_val) / (max_val - min_val)
    # y_valid = (y_valid - min_val) / (max_val - min_val)
    # y_test = (y_test - min_val) / (max_val - min_val)

    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train['Y/N'])
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    y_train = y_train['dropout_rate']
    y_valid = y_valid['dropout_rate']
    y_test = y_test['dropout_rate']

    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train.index.tolist()
    train_label_lst = y_train.values.tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_2_collate_fn)
    
    valid_nctid_lst = X_valid.index.tolist()
    valid_label_lst = y_valid.values.tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_2_collate_fn)

    test_nctid_lst = X_test.index.tolist()
    test_label_lst = y_test.values.tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_2_collate_fn)

    return train_loader, valid_loader, test_loader, 0, len(numerical_features + hot_features)

def patient_dropout_rate_yn(phase):
    # target = 'patient_dropout_rate'
    target = 'patient-dropout-event-forecasting'
    X_train, y_train, X_test, y_test = read_data(target, phase)
    y_train = y_train['Y/N']
    y_test = y_test['Y/N']

    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)

    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train.index.tolist()
    train_label_lst = y_train.to_numpy().tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_collate_fn)
    
    valid_nctid_lst = X_valid.index.tolist()
    valid_label_lst = y_valid.to_numpy().tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    test_nctid_lst = X_test.index.tolist()
    test_label_lst = y_test.to_numpy().tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    return train_loader, valid_loader, test_loader, 1, len(numerical_features + hot_features)



def duration(phase):
    # target = 'trial-duration-prediction'
    target = 'trial-duration-forecasting'
    X_train, y_train, X_test, y_test = read_data(target, phase)

    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)
        
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


    # maxmin scalar
    # min_val = 0
    # max_val = 10
    # y_train = (y_train - min_val) / (max_val - min_val)
    # y_valid = (y_valid - min_val) / (max_val - min_val)
    # y_test = (y_test - min_val) / (max_val - min_val)

    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    if isinstance(y_train, pd.DataFrame) or (hasattr(y_train, 'shape') and len(y_train.shape) > 1 and y_train.shape[1] > 1):
        cat_encoder.fit(X_train[cat_features], y_train.iloc[:, 0] if isinstance(y_train, pd.DataFrame) else y_train[:, 0])
    else:
        cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train.index.tolist()
    train_label_lst = y_train.to_numpy().tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_collate_fn)
    
    valid_nctid_lst = X_valid.index.tolist()
    valid_label_lst = y_valid.to_numpy().tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    test_nctid_lst = X_test.index.tolist()
    test_label_lst = y_test.to_numpy().tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    return train_loader, valid_loader, test_loader, 0, len(numerical_features + hot_features)

def outcome(phase):
    # target = 'trial-approval-prediction'
    target = 'trial-approval-forecasting'
    X_train, y_train, X_test, y_test = read_data(target, phase)
    y_train = y_train['outcome']
    y_test = y_test['outcome']
    
    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)
        
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train.index.tolist()
    train_label_lst = y_train.to_numpy().tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_collate_fn)
    
    valid_nctid_lst = X_valid.index.tolist()
    valid_label_lst = y_valid.to_numpy().tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    test_nctid_lst = X_test.index.tolist()
    test_label_lst = y_test.to_numpy().tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    return train_loader, valid_loader, test_loader, 1, len(numerical_features + hot_features)

def failure_reason(phase):
    def mapper(x):
        if x == 'poor enrollment':
            return 0
        elif x == 'efficacy':
            return 1
        elif x == 'safety':
            return 2
        elif x == 'Others':
            return 3
    # target = 'trial-failure-reason-prediction'
    target = 'trial-failure-reason-identification'
    X_train, y_train, X_test, y_test = read_data(target, phase)
    y_train = y_train['failure_reason'].apply(mapper)
    y_test = y_test['failure_reason'].apply(mapper)

    
    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)
        
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # 1、drop isna().sum() > len(data) * 0.5
    drop_columns = [c for c in X_train.columns if X_train[c].isna().sum() > len(X_train) * 0.5 and c != 'smiless' and c != 'icdcode']
    X_train = X_train.drop(columns=drop_columns, axis=1)
    X_valid = X_valid.drop(columns=drop_columns, axis=1)
    X_test = X_test.drop(columns=drop_columns, axis=1)


    text_feature = ['brief_title', 'brief_summary', 'detailed_description', 'eligibility/study_pop/textblock', 'intervention/description',
    'keyword', 'study_design_info/intervention_model_description', 'study_design_info/masking_description', 'condition', ]
    # condition -> condition_browse/mesh_term -> embedding
    # intervention/intervention_name -> intervention_browse/mesh_term -> embedding
    category_features = ['eligibility/gender', 'eligibility/healthy_volunteers', 'eligibility/sampling_method', 'has_expanded_access', 'oversight_info/has_dmc',
    'oversight_info/is_fda_regulated_device', 'oversight_info/is_fda_regulated_drug', 'patient_data/sharing_ipd', 'phase', 'responsible_party/responsible_party_type',
    'sponsors/lead_sponsor/agency_class', 'study_design_info/allocation', 'study_design_info/intervention_model', 'study_design_info/masking_num',
    'study_design_info/observational_model', 'study_design_info/primary_purpose', 'study_design_info/time_perspective', 'study_type'] 
    multihot_feature = [c for c in X_train.columns if "MaskingType-" in c or "ipd_info_type-" in c] # "MaskingType-*"(0/1) + ipd_info_type-*(0/1)
    int_feature = ['enrollment', 'number_of_arms'] + [c for c in X_train.columns if "Number" in c or 'masking_num' in c] # "*arm number", "intervention number"
    age_feature = ['eligibility/minimum_age', 'eligibility/maximum_age',]
    
    for c in age_feature:
        if c in X_train.columns:
            X_train[c] = X_train[c].apply(refine_year)
            X_valid[c] = X_valid[c].apply(refine_year)
            X_test[c] = X_test[c].apply(refine_year)

    # 2、category -> leave one out
    cat_features = []
    for c in category_features:
        if c in X_train.columns:
            cat_features.append(c)
            X_train[c].fillna(X_train[c].mode()[0], inplace=True)
            X_valid[c].fillna(X_valid[c].mode()[0], inplace=True)
            X_test[c].fillna(X_test[c].mode()[0], inplace=True)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_valid[cat_features] = cat_encoder.transform(X_valid[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])

    numerical_features = [c for c in X_train.columns if c in  cat_features + int_feature + age_feature]
    for c in numerical_features:
        X_train[c].fillna(X_train[c].mean(), inplace=True)
        X_valid[c].fillna(X_valid[c].mean(), inplace=True)
        X_test[c].fillna(X_test[c].mean(), inplace=True)

    text_features = [c for c in X_train.columns if c in text_feature]

    hot_features = [c for c in X_train.columns if c in multihot_feature]
    for c in hot_features:
        X_train[c].fillna(0, inplace=True)
        X_valid[c].fillna(0, inplace=True)
        X_test[c].fillna(0, inplace=True)

    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    # X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features] = quantile_transform(X_train[numerical_features], X_valid[numerical_features], X_test[numerical_features])
    train_nctid_lst = X_train.index.tolist()
    train_label_lst = y_train.to_numpy().tolist()
    train_icdcode_lst = X_train['icdcode'].fillna('["unknown"]').tolist()
    train_drugs_lst = X_train['intervention/intervention_name'].tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_criteria_lst = X_train['eligibility/criteria/textblock'].fillna("unknown").tolist()
    train_tabular_lst = X_train[numerical_features + hot_features].values.tolist()
    train_text_lst = X_train[text_features].values.tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Trial_Dataset_tabular(train_nctid_lst, train_label_lst, train_smiles_lst, train_icdcode_lst, train_criteria_lst, train_tabular_lst, train_text_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=trial_tabular_collate_fn)
    
    valid_nctid_lst = X_valid.index.tolist()
    valid_label_lst = y_valid.to_numpy().tolist()
    valid_icdcode_lst = X_valid['icdcode'].fillna('["unknown"]').tolist()
    valid_drugs_lst = X_valid['intervention/intervention_name'].tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_criteria_lst = X_valid['eligibility/criteria/textblock'].fillna("unknown").tolist()
    valid_tabular_lst = X_valid[numerical_features + hot_features].values.tolist()
    valid_text_lst = X_valid[text_features].values.tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Trial_Dataset_tabular(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_icdcode_lst, valid_criteria_lst, valid_tabular_lst, valid_text_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    test_nctid_lst = X_test.index.tolist()
    test_label_lst = y_test.to_numpy().tolist()
    test_icdcode_lst = X_test['icdcode'].fillna('["unknown"]').tolist()
    test_drugs_lst = X_test['intervention/intervention_name'].tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_criteria_lst = X_test['eligibility/criteria/textblock'].fillna("unknown").tolist()
    test_tabular_lst = X_test[numerical_features + hot_features].values.tolist()
    test_text_lst = X_test[text_features].values.tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Trial_Dataset_tabular(test_nctid_lst, test_label_lst, test_smiles_lst, test_icdcode_lst, test_criteria_lst, test_tabular_lst, test_text_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=trial_tabular_collate_fn)

    return train_loader, valid_loader, test_loader, 4, len(numerical_features + hot_features)


class Dose_dataset(Dataset):
    def __init__(self, nctid_lst, label_lst, smiles_lst,  mesh_lst):
        self.nctid_lst = nctid_lst
        self.label_lst = label_lst
        # self.label_name = [data[0] for data in label_lst]
        # self.label_max = [data[1] for data in label_lst]
        # self.label_min = [data[2] for data in label_lst]
        # self.label_avg = [data[3] for data in label_lst]
        self.smiles_lst = smiles_lst
        self.mesh_lst = mesh_lst
    
    def __len__(self):
        return len(self.nctid_lst)

    def __getitem__(self, idx):
        return self.nctid_lst[idx], self.label_lst[idx], self.smiles_lst[idx], self.mesh_lst[idx]

def dose(phase):
    target = 'drug-dose-prediction'
    X_train, y_train, X_test, y_test = read_data(target, phase)

    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    numerical_features = []
    text_features = ['smiless']
    hot_features = []
    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]

    def dose_collate_fn(x):
        nctid_lst = [i[0] for i in x]     ### ['NCT00604461', ..., 'NCT00788957'] 
        # label_vec = default_collate([i[1] for i in x])  ### shape n,
        label_name,label_max,label_min,label_avg = [], [], [], []
        label_data = [i[1] for i in x]
        label_name = [data[0] for data in label_data]
        label_max = [data[1] for data in label_data]
        label_min = [data[2] for data in label_data]
        label_avg = [data[3] for data in label_data]
        label_vec = default_collate([label_name, label_max, label_min, label_avg])
    
        smiles_lst = [smiles_txt_to_lst(i[2]) for i in x]
        mesh_lst = [mesh_term2feature(i[3]) for i in x]
        return [nctid_lst, label_vec, smiles_lst, mesh_lst]

    train_nctid_lst = X_train.index.tolist()
    train_label_lst = y_train.values.tolist()
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Dose_dataset(train_nctid_lst, train_label_lst, train_smiles_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=dose_collate_fn)
    
    valid_nctid_lst = X_valid.index.tolist()
    valid_label_lst = y_valid.values.tolist()
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Dose_dataset(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=dose_collate_fn)

    test_nctid_lst = X_test.index.tolist()
    test_label_lst = y_test.values.tolist()
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Dose_dataset(test_nctid_lst, test_label_lst, test_smiles_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=dose_collate_fn)

    return train_loader, valid_loader, test_loader, 2, 0


from importlib import resources
import socket
import time
def dose_cls(phase):
    target = 'drug-dose-prediction'

    X_train, y_train, X_test, y_test = read_data(target, phase)
    if 'nctid' not in X_train.columns:
        X_train.rename(columns={'ntcid': 'nctid'}, inplace=True)
        X_test.rename(columns={'ntcid': 'nctid'}, inplace=True)
    
    # Randomly split the training set into a validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    numerical_features = []
    text_features = ['smiless']
    hot_features = []
    mesh_term = [c for c in X_train.columns if 'mesh_term' in c]
    
    def dose_collate_fn(x):
        nctid_lst = [i[0] for i in x]     ### ['NCT00604461', ..., 'NCT00788957'] 
        # label_vec = default_collate([np.array(i[1]) for i in x])  ### shape n,
        label_name,label_max,label_min,label_avg = [], [], [], []
        label_data = [i[1] for i in x]
        label_name = [data[0] for data in label_data]
        label_max = [data[1] for data in label_data]
        label_min = [data[2] for data in label_data]
        label_avg = [data[3] for data in label_data]
        label_vec = default_collate([label_name, label_max, label_min, label_avg])
        
        smiles_lst = [smiles_txt_to_lst(i[2]) for i in x]
        mesh_lst = [mesh_term2feature(i[3]) for i in x]
        return [nctid_lst, label_vec, smiles_lst, mesh_lst]

    train_nctid_lst = X_train.index.tolist()
    train_label_lst = y_train.to_numpy().tolist()
    # train_label_lst = [i[1:] for i in train_label_lst]
    train_smiles_lst = X_train['smiless'].fillna('["unknown"]').tolist()
    train_mesh_lst = X_train[mesh_term].values.tolist()
    train_dataset = Dose_dataset(train_nctid_lst, train_label_lst, train_smiles_lst, train_mesh_lst)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=dose_collate_fn)
    
    valid_nctid_lst = X_valid.index.tolist()
    valid_label_lst = y_valid.to_numpy().tolist()
    # valid_label_lst = [i[1:] for i in valid_label_lst]
    valid_smiles_lst = X_valid['smiless'].fillna('["unknown"]').tolist()
    valid_mesh_lst = X_valid[mesh_term].values.tolist()
    valid_dataset = Dose_dataset(valid_nctid_lst, valid_label_lst, valid_smiles_lst, valid_mesh_lst)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=dose_collate_fn)

    test_nctid_lst = X_test.index.tolist()
    test_label_lst = y_test.to_numpy().tolist()
    # test_label_lst = [i[1:] for i in test_label_lst]
    test_smiles_lst = X_test['smiless'].fillna('["unknown"]').tolist()
    test_mesh_lst = X_test[mesh_term].values.tolist()
    test_dataset = Dose_dataset(test_nctid_lst, test_label_lst, test_smiles_lst, test_mesh_lst)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=dose_collate_fn)

    return train_loader, valid_loader, test_loader, 4, 0



    