import pandas as pd
from tqdm import tqdm
from utils import Dictionary

train_dataset_filename = "result/simplifyweibo_4_moods/train.csv"
test_dataset_filename = "result/simplifyweibo_4_moods/test.csv"
dict_file_name = "result/simplifyweibo_4_moods/dictionary.json"
data_dict = Dictionary()


save_dataset = pd.read_csv(train_dataset_filename)
dataset_length = len(save_dataset)
for idx in tqdm(range(dataset_length),desc="Building dictionary"):
        for word in save_dataset.loc[idx,"review"].split("\t"):
            data_dict.add(word)
save_dataset = pd.read_csv(test_dataset_filename)
dataset_length = len(save_dataset)
for idx in tqdm(range(dataset_length),desc="Building dictionary"):
        for word in save_dataset.loc[idx,"review"].split("\t"):
            data_dict.add(word)


data_dict.save(dict_file_name)