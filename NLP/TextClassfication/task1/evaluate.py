import os
import time
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import f1_score,accuracy_score,jaccard_score
from sklearn.metrics import confusion_matrix
from utils import to_device

class Evaluator:
    def __init__(self):
        self.data_dict = {
            "f1-score":[],
            "loss-score":[],
            "acc-score":[],
            "jac-score":[],
            "time-record":[]
        }
        self.corr_list = []
    def evaluate(self,model,datalaoder,device):
        model.eval()
        target_list = []
        predict_list = []
        for item in datalaoder:
            item = to_device(item,device)
            input_tensor = item[0]
            target_label = item[1]
            probability = model(input_tensor)
            target = target_label.detach().cpu().numpy()
            predict = torch.argmax(probability,dim=1).detach().cpu().numpy()
            target_list.append(target)
            predict_list.append(predict)
        y_true = np.hstack(target_list)
        y_pred = np.hstack(predict_list)
        corr = confusion_matrix(y_true,y_pred)
        self.corr_list.append(corr)
        jac_score_value = jaccard_score(y_true, y_pred, average='macro')
        acc_score_value = accuracy_score(y_true, y_pred)
        f1_score_value = f1_score(y_true, y_pred, average='macro')
        self.data_dict["f1-score"].append(f1_score_value)
        self.data_dict["jac-score"].append(jac_score_value)
        self.data_dict["acc-score"].append(acc_score_value)
        return corr,f1_score_value,acc_score_value,jac_score_value
    def add_loss(self,loss):
        self.data_dict["loss-score"].append(loss)
    def save(self,save_filename):
        save_df = pd.DataFrame()
        for key in self.data_dict:
            save_df[key] = self.data_dict[key]
        save_df.to_csv(save_filename,index=None)
    def draw(self,model_name,save_path):
        pass
    def begin_time(self):
        self.time_step = time.time()
    def end_time(self):
        time_delta = time.time() - self.time_step
        self.data_dict["time-record"].append(time_delta)
    

    