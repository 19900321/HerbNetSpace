import argparse
import pickle
import numpy as np
import pandas as pd
import tqdm
import os
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
from utils import set_seed
from Evaluation_index import DiseaseDrugRecall
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_pred, y_true)
    spearman_corr, _ = spearmanr(y_pred, y_true)
    return mse, rmse, mae, r2, pearson_corr, spearman_corr

def split_dataset(pairf, embeddingf,validr, testr, seed):
    embedding_dict = {}
    if os.path.exists(embeddingf):
        print(f"File exists: {embeddingf}")
    with open(embeddingf, 'rb') as fin:
        embedding_dict = pickle.load(fin)

    xs, ys,  = [], []
    try:
        with open(pairf, 'r') as fin:
            lines = fin.readlines()
    except Exception as e:
        # print(f"Error reading pair file: {e}")
        return [], []

        
    for line in lines[1:]:
        line = line.strip().split(',')
        # drug1,drug2,dis,label = line
        # print(line)
        drug1 = line[0]
        drug2 = line[1]
        dis = line[2]
        label = line[3]
        if drug1 not in embedding_dict or drug2 not in embedding_dict or dis not in embedding_dict:
            continue
        xs.append(embedding_dict[drug1] + embedding_dict[drug2] - embedding_dict[dis])
        ys.append(float(label))
        
    # dataset split
    x, y = {}, {}
    x['train'], x['test'], y['train'], y['test'] = train_test_split( 
        xs, ys, test_size=testr, random_state=seed)
    if validr > 0:
        x['train'], x['valid'], y['train'], y['valid'] = train_test_split( x['train'], y['train'],test_size=validr/(1-testr), 
            random_state=seed)
    else:
        x['valid'], y['valid'] = [], []
    return x, y




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


def predict_dda(embeddingf:str, pairf:str, modelf:str='clf.pkl', seed:int=42,
                validr:float=0.1, testr:float=0.1):
    set_seed(seed)
    x,y = split_dataset(pairf, embeddingf, validr, testr, seed)
    
    clf = XGBRegressor(base_score = 0.5, booster = 'gblinear',eval_metric ='error',objective = 'reg:squarederror',
        gamma = 0,learning_rate = 0.1, max_depth =6,n_estimators = 500,
        tree_method = 'auto',min_child_weight = 4,subsample = 0.8, colsample_bytree = 0.9,
        scale_pos_weight = 1,max_delta_step = 1,seed = seed) 
    
    clf.fit(x['train'], y['train'])
    
    preds = {}
    result_list = []
    for split in ['train','valid','test']:
        set = split.upper()[0:5],
        preds[split] = clf.predict(np.array(x[split]))
        if split == 'test':
            zipped_list = list(zip(preds[split],y[split]))
            Df = pd.DataFrame(zipped_list,columns = ['prob','label'])
    
    
    if not os.path.exists(os.path.dirname(modelf)):
        os.mkdir(os.path.dirname(modelf))
    with open(modelf,'wb') as fw:
        pickle.dump(clf, fw)

        
    return Df




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
    

    yin_mpl(256, data.x_train, data.y_train,data.x_test,data.y_test)
    

          
if __name__ == '__main__':
    args=parse_args()
    predict_dda(**args)

def Predict_dda(embeddingf:str, pairf:str, modelf:str='clf.pkl', seed:int=42,
                validr:float=0.1, testr:float=0.1):
    set_seed(seed)
    X,y = Split_dataset(pairf, embeddingf)

        
    return X,y

def Split_dataset(pairf, embeddingf):
    # with open(embeddingf, 'rb') as fin:
    #     embedding_dict = pickle.load(fin)
    embedding_dict = {}
    if os.path.exists(embeddingf):
        print(f"File exists: {embeddingf}")
    with open(embeddingf, 'rb') as fin:
        embedding_dict = pickle.load(fin)

    xs, ys,  = [], []
    try:
        with open(pairf, 'r') as fin:
            lines = fin.readlines()
    except Exception as e:
        print(f"Error reading pair file: {e}")
        return [], []

        
    for line in lines[1:]:
        line = line.strip().split(',')
        # drug1,drug2,dis,label = line
        print(line)
        drug1 = line[0]
        drug2 = line[1]
        dis = line[2]
        label = line[3]
        if drug1 not in embedding_dict or drug2 not in embedding_dict or dis not in embedding_dict:
            continue
        xs.append(embedding_dict[drug1] + embedding_dict[drug2] - embedding_dict[dis])
        ys.append(float(label))
        
    return xs, ys