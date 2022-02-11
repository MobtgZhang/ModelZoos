import os
import json
import sklearn
import ltp
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
def to_device(item,device):
    item_new = []
    for ex in item:
        item_new.append(ex.to(device))
    return item_new
def vectorize(data_item,data_dict,max_seq_len):
    label = data_item['label']
    sentence = data_item['review'].split("\t")
    seg_sent = [data_dict[word] for word in sentence]
    if len(seg_sent)>max_seq_len:
        seg_sent = seg_sent[:max_seq_len]
    else:
        seg_sent += [data_dict['<PAD>']]*(max_seq_len-len(seg_sent))
    seg_sent = [data_dict['<START>']] + seg_sent + [data_dict['<END>']]
    return seg_sent,label
def bacthfy(batch):
    seg_ids = [ex[0] for ex in batch]
    labels = [ex[1] for ex in batch]
    return torch.tensor(seg_ids,dtype=torch.long),torch.tensor(labels,dtype=torch.long)
class LabelReviewSplitWordsDataset(Dataset):
    def __init__(self,load_file_name,data_dict_file,max_seq_len = 64):
        super(LabelReviewSplitWordsDataset,self).__init__()
        self.load_file_name = load_file_name
        self.max_seq_len = max_seq_len
        self.data_dict_file = data_dict_file
        self.data_dict = Dictionary.load(data_dict_file)
        self.dataset = pd.read_csv(load_file_name)
        self.seq_mean_len = sum(self.dataset['review'].map(lambda x:len(x.split("\t"))))/len(self)
    def __getitem__(self,idx):
        return vectorize(self.dataset.iloc[idx,:],self.data_dict,self.max_seq_len)
    def __len__(self):
        return len(self.dataset)
class Dictionary:
    def __init__(self):
        self.name = 'default'
        self.ind2token = ['<PAD>','<START>','<END>','<UNK>',]
        self.token2ind = {'<PAD>':0,'<START>':1,'<END>':2,'<UNK>':3}
        self.start_index = 0
        self.end_index = len(self.ind2token)
    def __iter__(self):
        return self
    def __next__(self):
        if self.start_index < self.end_index:
            ret = self.ind2token[self.start_index]
            self.start_index += 1
            return ret
        else:
            raise StopIteration
    def __getitem__(self,item):
        if type(item) == str:
            return self.token2ind.get(item,"<UNK>")
        elif type(item) == int:
            word = self.ind2token[item]
            return word
        else:
            raise IndexError()
    def add(self,word):
        if word not in self.token2ind:
            self.token2ind[word] = len(self.ind2token)
            self.ind2token.append(word)
            self.end_index = len(self.ind2token)
    def save(self,save_file):
        with open(save_file,"w",encoding="utf-8") as wfp:
            data = {
                "ind2token":self.ind2token,
                "token2ind":self.token2ind,
            }
            json.dump(data,wfp)
    @staticmethod
    def load(load_file):
        tp_dict = Dictionary()
        with open(load_file,"r",encoding="utf-8") as rfp:
            data = json.load(rfp)
            tp_dict.token2ind = data["token2ind"]
            tp_dict.ind2token = data["ind2token"]
            tp_dict.end_index = len(tp_dict.ind2token)
        return tp_dict
    def __contains__(self,word):
        assert type(word) == str
        return word in self.token2ind
    def __len__(self):
        return len(self.token2ind)
    def __repr__(self) -> str:
        return '{}(num_keys={})'.format(
            self.__class__.__name__,len(self.token2ind))
    def __str__(self) -> str:
        return '{}(num_keys={})'.format(
            self.__class__.__name__,len(self.token2ind))
def build_simplifyweibo_4_moods_dataset(data_path,result_path,percentage = 0.75,batch_size=160):
    raw_file_name = os.path.join(data_path,"simplifyweibo_4_moods.csv")
    train_file_name = os.path.join(result_path,"train.csv")
    test_file_name = os.path.join(result_path,"test.csv")
    dict_file_name = os.path.join(result_path,"dictionary.json")
    dataset = pd.read_csv(raw_file_name)
    save_dataset = dataset.copy()
    data_dict = Dictionary()
    m_ltp = ltp.LTP()
    dataset_length = len(save_dataset)
    btx_len = dataset_length//batch_size + 1
    for btx in tqdm(range(btx_len),desc="Spliting words"):
        tp_sents = dataset.loc[btx*batch_size:(btx+1)*batch_size,"review"].values.tolist()
        segment_sents, _ = m_ltp.seg(tp_sents)
        segment_sents = ["\t".join(sent) for sent in segment_sents]
        save_dataset.loc[btx*batch_size:(btx+1)*batch_size,"review"] = segment_sents
    for idx in tqdm(range(dataset_length),desc="Building dictionary"):
        for word in save_dataset.loc[idx,"review"].split("\t"):
            data_dict.add(word)
    data_dict.save(dict_file_name)
    save_dataset = sklearn.utils.shuffle(save_dataset)
    save_dataset = save_dataset.reindex()
    save_dataset = save_dataset.dropna(axis=0, how='any')
    data_len = len(save_dataset)
    train_len = int(percentage*data_len)
    train_dataset = save_dataset.iloc[:train_len,:]
    test_dataset = save_dataset.iloc[train_len:,:]
    train_dataset.to_csv(train_file_name,index=None)
    test_dataset.to_csv(test_file_name,index=None)
