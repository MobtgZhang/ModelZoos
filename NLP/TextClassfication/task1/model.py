import torch
import torch.nn.functional as F
import torch.nn as nn
class TextCNN(nn.Module):
    def __init__(self,n_class,vocab_size=5000,embedding_dim=300,num_filters=64,
                    filters =(2,3,4),dropout_rate = 0.15,max_seq_len=256):
        super(TextCNN,self).__init__()
        self.n_class = n_class
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.filters = filters
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.word_embeddings = nn.Embedding(vocab_size,embedding_dim)
        
        # Convolutional Layer
        self.conv_list = nn.ModuleList([nn.Conv1d(embedding_dim,num_filters,num) for num in filters])
        # Maxpooling Layer
        self.max_pool = nn.ModuleList([nn.MaxPool1d(max_seq_len-num+1) for num in filters])
        # Dropout Layer
        self.dropout = nn.Dropout(dropout_rate)
        # classification Layer
        lin_dim = len(filters)*num_filters
        self.fc = nn.Linear(lin_dim,n_class)
    @staticmethod 
    def load_embedding(pretrain_file_name,max_seq_len=256):
        model = TextCNN()
        model.word_embeddings = nn.Embedding.from_pretrain(pretrain_file_name,freeze=False)
        model.vocab_size = model.word_embeddings.num_embeddings
        model.embedding_dim = model.word_embeddings.embedding_dim
        return model
    def forward(self,in_ids):
        """
        in_ids: (batch_size,seq_len)
        """
        embed_in = self.word_embeddings(in_ids) # (batch_size,seq_len,embedding_dim)
        embed_in = embed_in.transpose(2,1).contiguous() # (batch_size,seq_len,embedding_dim)

        conv_out = [max_pool(F.relu(conv(embed_in))) 
                    for conv,max_pool in zip(self.conv_list,self.max_pool)]
        out_put = torch.cat(conv_out,dim=1)
        out_put = self.dropout(out_put).squeeze()
        out_put = self.fc(out_put)
        return F.log_softmax(out_put,dim=1)

