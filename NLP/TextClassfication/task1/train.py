import os
import logging
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import build_simplifyweibo_4_moods_dataset
from utils import LabelReviewSplitWordsDataset
from utils import bacthfy,to_device
from model import TextCNN, TextRNN, RCNN,CapsAttNet,DPCNN
from config import get_args,check_args
from evaluate import Evaluator

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def train_simplifyweibo_4_moods(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # preparing the dataset
    data_path = os.path.join(args.data_dir,args.dataset)
    result_path = os.path.join(args.result_dir,args.dataset)
    train_file_name = os.path.join(args.result_dir,args.dataset,"train.csv")
    test_file_name = os.path.join(args.result_dir,args.dataset,"test.csv")
    dict_file_name = os.path.join(args.result_dir,args.dataset,"dictionary.json")
    if not os.path.exists(train_file_name) or \
        not os.path.exists(test_file_name) or \
            not os.path.exists(dict_file_name):
        # create the dataset
        build_simplifyweibo_4_moods_dataset(data_path,result_path)
    train_dataset = LabelReviewSplitWordsDataset(train_file_name,dict_file_name,max_seq_len=args.max_seq_len)
    train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=bacthfy,num_workers= args.num_workers)
    test_dataset = LabelReviewSplitWordsDataset(test_file_name,dict_file_name,max_seq_len=args.max_seq_len)
    test_dataloader = DataLoader(test_dataset,batch_size=args.test_batch_size,shuffle=True,collate_fn=bacthfy,num_workers= args.num_workers)

    # preparing the model 
    if args.model.lower() == "textcnn":
        textcnn_yaml = os.path.join(args.config_dir,"TextCNN.yaml")
        with open(textcnn_yaml,mode="r",encoding="utf-8") as rfp:
            kwargs = yaml.safe_load(rfp.read())
        model = TextCNN(args.n_class,vocab_size=len(train_dataset.data_dict),**kwargs)
    elif args.model.lower() == "textrnn":
        textcnn_yaml = os.path.join(args.config,"TextRNN.yaml")
        with open(textcnn_yaml,mode="r",encoding="utf-8") as rfp:
            kwargs = yaml.safe_load(rfp.read())
        model = TextRNN(args.n_class,vocab_size=len(train_dataset.data_dict),**kwargs)
    elif args.model.lower() == "rcnn":
        rcnn_yaml = os.path.join(args.config_dir,"RCNN.yaml")
        with open(rcnn_yaml,mode="r",encoding="utf-8") as rfp:
            kwargs = yaml.safe_load(rfp.read())
        model = RCNN(args.n_class,vocab_size=len(train_dataset.data_dict),**kwargs)
    elif args.model.lower() == "capsattnet":
        capsattnet_yaml = os.path.join(args.config_dir,"CapsAttNet.yaml")
        with open(capsattnet_yaml,mode="r",encoding="utf-8") as rfp:
            kwargs = yaml.safe_load(rfp.read())
        model = CapsAttNet(args.n_class,vocab_size=len(train_dataset.data_dict),**kwargs)
    elif args.model.lower() == "dpcnn":
        dpcnn_yaml = os.path.join(args.config_dir,"DPCNN.yaml")
        with open(dpcnn_yaml,mode="r",encoding="utf-8") as rfp:
            kwargs = yaml.safe_load(rfp.read())
        model = DPCNN(args.n_class,vocab_size=len(train_dataset.data_dict),**kwargs)
    else:
        raise ValueError("Unknown model name %s"%args.model)
    kwargs["max_seq_len"] = args.max_seq_len
    args.learning_rate = kwargs["learning_rate"]
    model.to(device)
    optimizer = optim.Adamax(model.parameters(),lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    evaluator = Evaluator()
    for epoch in range(args.epoch_times):
        loss_all = 0.0
        model.train()
        evaluator.begin_time()
        for item in train_dataloader:
            optimizer.zero_grad()
            item = to_device(item,device)
            input_tensor = item[0]
            target_label = item[1]
            predict_probability = model(input_tensor)
            loss = loss_fn(predict_probability,target_label)
            loss_all += loss.cpu().detach().numpy()
            loss.backward()
            optimizer.step()
        loss_all /= len(train_dataloader)
        corr,f1_score_value,acc_score_value,jac_score_value = evaluator.evaluate(model,test_dataloader,device)
        evaluator.add_loss(loss_all)
        logger.info("Epoches %d, complete!, avg loss %0.4f,f1-score %0.4f,accuracy-score %0.4f,jaccard-score %0.4f."%(epoch + 1,loss_all,f1_score_value,acc_score_value,jac_score_value))
        loss_all/=len(train_dataloader)
        evaluator.end_time()
    evaluator.draw(args.model_name,result_path)
    evaluate_file_name = os.path.join(result_path,args.model_name+"_results.csv")
    evaluator.save(evaluate_file_name)
def main():
    args = get_args()
    check_args(args)
    # First ,create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log level switch
    # Second, create a handler ,which is used for writing log files
    logfile = os.path.join(args.log_dir,args.model_name + '.log')
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)  
    # Third，define the output format for handler
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    # Fourth，add loggerin the handler
    logger.addHandler(fh)
    logger.info(str(args))
    if args.dataset == "simplifyweibo_4_moods":
        args.n_class = 4
        train_simplifyweibo_4_moods(args)
if __name__ == "__main__":
    main()

