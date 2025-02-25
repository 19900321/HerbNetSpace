import argparse
import pickle
import numpy as np
import pandas as pd
import tqdm
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.nn import Linear
import torch.nn.functional as F
from DREAMwalk.utils import set_seed
from DREAMwalk.Evaluation_index import DiseaseDrugRecall
from sklearn.svm import SVC

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', type=str, required=True)
    parser.add_argument('--pair_file', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--model_checkpoint', type=str, default='clf.pkl')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    
    args = parser.parse_args()
    args = {'embeddingf':args.embedding_file,
     'pairf':args.pair_file,
     'seed':args.seed,
     'patience':args.patience,
     'modelf':args.model_checkpoint,
     'testr':args.test_ratio,
     'validr':args.valid_ratio
     }
    return args



def split_dataset(pairf, embeddingf, validr, testr, seed):
    embedding_dict = {}
    if os.path.exists(embeddingf):
        print(f"File exists: {embeddingf}")
    with open(embeddingf, 'rb') as fin:
        embedding_dict = pickle.load(fin)

    xs, ys, ids = [], [], []
    try:
        with open(pairf, 'r') as fin:
            lines = fin.readlines()
    except Exception as e:
        print(f"Error reading pair file: {e}")
        return [], [], []

        
    for line in lines[1:]:
        line = line.strip().split('\t')
        print(line)
        id = line[0]
        drug = line[1]
        dis = line[2]
        label = line[3]
        xs.append(embedding_dict[drug] - embedding_dict[dis])
        ys.append(int(label))
        ids.append(id)  # 记录编号
        
    # dataset split
    x, y,id_dict = {}, {}, {}
    x['train'], x['test'], y['train'], y['test'],id_dict['train'],id_dict['test'] = train_test_split( 
        xs, ys, ids, test_size=testr, random_state=seed, stratify=ys)
    if validr > 0:
        x['train'], x['valid'], y['train'], y['valid'],id_dict['train'], id_dict['valid'] = train_test_split( x['train'], y['train'], id_dict['train'],test_size=validr/(1-testr), 
            random_state=seed, stratify=y['train'])
    else:
        x['valid'], y['valid'] = [], []
        id_dict['vaild'] = []
    return x, y,id_dict


def split_dataset_no_vaild(pairf, embeddingf, testr, seed):
    embedding_dict = {}
    if os.path.exists(embeddingf):
        print(f"File exists: {embeddingf}")
    with open(embeddingf, 'rb') as fin:
        embedding_dict = pickle.load(fin)

    xs, ys, ids = [], [], []
    try:
        with open(pairf, 'r') as fin:
            lines = fin.readlines()
    except Exception as e:
        return [], [], []

    for line in lines[1:]:
        line = line.strip().split('\t')
        id = line[0]
        drug = line[1]
        dis = line[2]
        label = line[3]
        xs.append(embedding_dict[drug] - embedding_dict[dis])
        ys.append(int(label))
        ids.append(id)  # 记录编号
        
    # dataset split
    x, y,id_dict = {}, {}, {}
    x['train'], x['test'], y['train'], y['test'],id_dict['train'],id_dict['test'] = train_test_split( 
        xs, ys, ids, test_size=testr, random_state=seed, stratify=ys)
    return x, y,id_dict


def return_scores(target_list, pred_list):
    metric_list = [
        accuracy_score, 
        roc_auc_score, 
        average_precision_score, 
        f1_score
    ] 
    
    scores = []
    for metric in metric_list:
        if metric in [roc_auc_score, average_precision_score]:
            scores.append(metric(target_list,pred_list))
        else: # accuracy_score, f1_score
            scores.append(metric(target_list, pred_list.round())) 
    return scores


def predict_dda(embeddingf:str, pairf:str, modelf:str='clf.pkl', seed:int=42, testr:float=0.3):
    set_seed(seed)
    x,y,id_dict = split_dataset_no_vaild(pairf, embeddingf, testr, seed)
    
    clf = XGBClassifier(base_score = 0.5, booster = 'gbtree',eval_metric ='error',objective = 'binary:logistic',
        gamma = 0,learning_rate = 0.1, max_depth =6,n_estimators = 500,
        tree_method = 'auto',min_child_weight = 4,subsample = 0.8, colsample_bytree = 0.9,
        scale_pos_weight = 1,max_delta_step = 1,seed = seed) 
    
    clf.fit(x['train'], y['train'])
    
    preds = {}
    scores = {}
    result_list = []
    for split in ['train','test']:
        set = split.upper()[0:5],
        preds[split] = clf.predict_proba(np.array(x[split]))[:, 1]
        if split == 'test':
            zipped_list = list(zip(preds[split],y[split],id_dict[split]))
            Df = pd.DataFrame(zipped_list,columns = ['prob','label','id'])
    
    if not os.path.exists(os.path.dirname(modelf)):
        os.mkdir(os.path.dirname(modelf))
    with open(modelf,'wb') as fw:
        pickle.dump(clf, fw)
    return Df


def MLP_predict_dda(embeddingf: str, pairf: str, seed: int = 42,testr: float = 0.3):
    set_seed(seed)
    x, y, id_dict = split_dataset_no_vaild(pairf, embeddingf, testr, seed)
    
    clf = MLPClassifier(
        hidden_layer_sizes=(100,),  # 隐藏层大小
        activation='relu',          # 激活函数
        solver='adam',              # 优化器
        alpha=0.0001,               # L2惩罚（正则化项）的参数
        learning_rate='adaptive',   # 学习率调节
        max_iter=500,               
        random_state=seed           
    )
    clf.fit(x['train'], y['train'])
    preds = {}
    for split in ['train','test']:
        preds[split] = clf.predict_proba(np.array(x[split]))[:, 1]
        if split == 'test':
            zipped_list = list(zip(preds[split], y[split], id_dict[split]))
            df = pd.DataFrame(zipped_list, columns=['prob', 'label', 'id'])
    return df


def SVM_predict_dda(embeddingf: str, pairf: str, seed: int = 42, testr: float = 0.3,model_path: str = 'svm_model.pkl'):
    set_seed(seed)
    
    x, y, id_dict = split_dataset_no_vaild(pairf, embeddingf, testr, seed)
    df1 = pd.DataFrame(x['train'])
    X1 = df1.values
    scaler = StandardScaler()
    x1_scaled = scaler.fit_transform(X1)

    df2 = pd.DataFrame(x['test'])
    X2 = df2.values
    scaler = StandardScaler()
    x2_scaled = scaler.fit_transform(X2)
    clf = SVC(probability=True, kernel='rbf', C=1.0, gamma=0.1,random_state=seed)
    clf.fit(x1_scaled, y['train'])
    
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {model_path}")
    if hasattr(clf, 'predict_proba'):
        print("Model has predict_proba method.")
    else:
        print("Model does not have predict_proba method.")
        
    preds = clf.predict_proba(np.array(x2_scaled))[:, 1]
    zipped_list = list(zip(preds, y['test'], id_dict['test']))
    df = pd.DataFrame(zipped_list, columns=['prob', 'label', 'id'])
    return df



def RF_predict_dda(embeddingf: str, pairf: str, seed: int = 42, testr: float = 0.3):
    set_seed(seed)
    x, y, id_dict = split_dataset_no_vaild(pairf, embeddingf, testr, seed)
    
    clf = RandomForestClassifier(
        n_estimators=100,  # Number of trees in the forest
        random_state=seed  # Control the randomness
    )
    clf.fit(x['train'], y['train'])
    preds = {}
    for split in ['train', 'test']:
        preds[split] = clf.predict_proba(np.array(x[split]))[:, 1]
        if split == 'test':
            zipped_list = list(zip(preds[split], y[split], id_dict[split]))
            df = pd.DataFrame(zipped_list, columns=['prob', 'label', 'id'])
    return df

def KNN_predict_dda(embeddingf: str, pairf: str, seed: int = 42,testr: float = 0.3):
    set_seed(seed)
    x, y, id_dict = split_dataset_no_vaild(pairf, embeddingf,testr, seed)
    
    clf = KNeighborsClassifier(
        n_neighbors=5,  # Number of neighbors to use
        weights='uniform',  # Weight function used in prediction
        algorithm='auto',  # Algorithm used to compute the nearest neighbors
        n_jobs=-1  # Number of parallel jobs to run for neighbors search
    )
    clf.fit(x['train'], y['train'])
    # with open(model_path,'wb') as f:
    #     pickle.dump(clf,f)
    preds = {}
    for split in ['train', 'test']:
        preds[split] = clf.predict_proba(np.array(x[split]))[:, 1]
        if split == 'test':
            zipped_list = list(zip(preds[split], y[split], id_dict[split]))
            df = pd.DataFrame(zipped_list, columns=['prob', 'label', 'id'])
    return df

def GBM_predict_dda(embeddingf: str, pairf: str, seed: int = 42, testr: float = 0.3):
    set_seed(seed)
    x, y, id_dict = split_dataset_no_vaild(pairf, embeddingf,testr, seed)
    
    clf = GradientBoostingClassifier(
        n_estimators=100,  # The number of boosting stages to be run
        learning_rate=0.1,  # Learning rate shrinks the contribution of each tree
        max_depth=3,  # Maximum depth of the individual regression estimators
        random_state=seed  # Control the randomness
    )
    clf.fit(x['train'], y['train'])
    preds = {}
    for split in ['train', 'test']:
        preds[split] = clf.predict_proba(np.array(x[split]))[:, 1]
        if split == 'test':
            zipped_list = list(zip(preds[split], y[split], id_dict[split]))
            df = pd.DataFrame(zipped_list, columns=['prob', 'label', 'id'])
    return df




#buil simple neural model
import torch
import torch.nn as nn

class simple_mpl(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(simple_mpl, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, int(hidden_size / 4))  # Convert to integer
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(int(hidden_size / 4), 1)  # Convert to integer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        output = self.sigmoid(x)
        return output

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Assuming you have your data loaded as features and targets (0 or 1)
    
class Creat_data():
    def __init__(self, embeddingf:str, pairf:str, seed:int=42,validr:float=0.2, testr:float=0.01):
        self.embeddingf = embeddingf
        self.pairf = pairf
        self.seed = seed
        self.validr = validr
        self.testr = testr
        self.x_train,  self.y_train, self.x_test,self.y_test, self.feature_size = self.get_split_data()


    # Split data into train and test sets
    def get_split_data(self):
        set_seed(self.seed)
        x,y = split_dataset(self.pairf, self.embeddingf, self.validr, self.testr, self.seed)
        input_size = pd.DataFrame(x['train']).shape[1]
        # Initialize the model

        features_array = np.array(x['train'])
        targets_array = np.array(y['train'])
        # Convert NumPy arrays to PyTorch tensors
        features_tensor = torch.tensor(features_array).float()  # Assuming features are floats
        targets_tensor = torch.tensor(targets_array).float() 

        features_array_test = np.array(x['train'])
        targets_array_test = np.array(y['train'])
        # Convert NumPy arrays to PyTorch tensors
        features_tensor_test = torch.tensor(features_array_test).float()  # Assuming features are floats
        targets_tensor_test = torch.tensor(targets_array_test).float()

        return features_tensor,  targets_tensor, features_tensor_test,targets_tensor_test, input_size


## predict by neural network model
def predict_drug_disease(embeddingf:str, pairf:str, modelf:str='clf_nn.pkl', seed:int=42,
                validr:float=0.2, testr:float=0.01):
       
    data = Creat_data(embeddingf, pairf, seed,validr, testr)
    input_size = data.feature_size
    #mlp = MLP(input_size, 16, 1)
    #mlp.fit(data, epochs=150)

    yin_mpl(256, data.x_train, data.y_train,data.x_test,data.y_test)



          
if __name__ == '__main__':
    args=parse_args()
    predict_dda(**args)