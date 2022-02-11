import argparse
def get_args():
    parser = argparse.ArgumentParser(description="This is the text classification demo.")
    parser.add_argument("--data-dir",type=str,default="./dataset",help="raw data path.")
    parser.add_argument("--result-dir",type=str,default="./result",help="result data path.")
    parser.add_argument("--log-dir",type=str,default="./result",help="log data path.")
    parser.add_argument("--dataset",type=str,default="simplifyweibo_4_moods",help="dataset name.")
    parser.add_argument("--percentage",type=float,default=0.75,help="the percentage of the dataset.")
    parser.add_argument("--max-seq-len",type=int,default=56,help="the max length of the sentence.")
    parser.add_argument("--test-batch-size",type=int,default=4,help="the test batch size of the dataset.")
    parser.add_argument("--batch-size",type=int,default=8,help="the batch size of the dataset.")
    parser.add_argument("--num-workers",type=int,default=4,help="the number workers of the dataset.")
    parser.add_argument("--embedding-dim",type=int,default=300,help="the number embedding dimension of the dataset.")
    parser.add_argument("--num-filters",type=int,default=32,help="the number filters of the model.")
    parser.add_argument("--epoch-times",type=int,default=10,help="the training times of the model.")
    parser.add_argument("--learning-rate",type=float,default=1e-4,help="the learning rate of the model.")
    parser.add_argument("--cuda", action='store_false', help="Training model by cuda.")
    args = parser.parse_args()
    return args
