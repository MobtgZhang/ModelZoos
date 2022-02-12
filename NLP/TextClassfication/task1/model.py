import torch
import torch.nn.functional as F
import torch.nn as nn

from layers import CapsuleLayer,SelfAttLayer

class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()
    def forward(self,x):
        x = x * torch.tanh(F.softplus(x))
        return x
class TextCNN(nn.Module):
    def __init__(self,n_class,**kwargs):
        super(TextCNN,self).__init__()
        self.config = {
            "n_class":n_class,
            "vocab_size":5000,
            "embedding_dim":300,
            "num_filters":64,
            "filters":(2,3,4),
            "dropout_rate":0.15,
            "max_seq_len":256
        }
        for key in kwargs:
            if key in self.config:
                self.config[key] = kwargs[key]
        self.word_embeddings = nn.Embedding(self.config["vocab_size"],self.config["embedding_dim"])
        # Convolutional Layer
        self.conv_list = nn.ModuleList([nn.Conv1d(self.config["embedding_dim"],self.config["num_filters"],num) for num in self.config["filters"]])
        # Maxpooling Layer
        self.max_pool = nn.ModuleList([nn.MaxPool1d(self.config["max_seq_len"]-num+1) for num in self.config["filters"]])
        # Dropout Layer
        self.dropout = nn.Dropout(self.config["dropout_rate"])
        # classification Layer
        lin_dim = len(self.config["filters"])*self.config["num_filters"]
        self.mish = Mish()
        self.fc = nn.Linear(lin_dim,n_class)
    @staticmethod 
    def load_embedding(pretrain_file_name,n_class):
        model = TextCNN(n_class)
        model.word_embeddings = nn.Embedding.from_pretrain(pretrain_file_name,freeze=False)
        model.config["vocab_size"] = model.word_embeddings.num_embeddings
        model.config["embedding_dim"] = model.word_embeddings.embedding_dim
        return model
    def forward(self,in_ids):
        """
        in_ids: (batch_size,seq_len)
        """
        embed_in = self.word_embeddings(in_ids) # (batch_size,seq_len,embedding_dim)
        embed_in = embed_in.transpose(2,1).contiguous() # (batch_size,embedding_dim,seq_len)

        conv_out = [max_pool(self.mish(conv(embed_in))) for conv,max_pool,num in zip(self.conv_list,self.max_pool,self.config["filters"])]
        out_put = torch.cat(conv_out,dim=1)
        out_put = self.dropout(out_put).squeeze()
        out_put = self.fc(out_put)
        return F.log_softmax(out_put,dim=1)
class TextRNN(nn.Module):
    def __init__(self,n_class,**kwargs):
        super(TextRNN,self).__init__()
        self.config = {
            "n_class":n_class,
            "vocab_size":5000,
            "embedding_dim":300,
            "hidden_dim":100,
            "num_layers":1,
            "bidirectional":True,
            "rnn_type":"lstm"
        }
        for key in kwargs:
            if key in self.config:
                self.config[key] = kwargs[key]
        self.config["rnn_type"] = self.config["rnn_type"].lower()
        assert self.config["rnn_type"] in ("lstm","gru","stack_lstm","sru")
        self.word_embeddings = nn.Embedding(self.config["vocab_size"],self.config["embedding_dim"])
        if self.config["rnn_type"]=="lstm":
            self.rnn = nn.LSTM(self.config["embedding_dim"],self.config["hidden_dim"],num_layers=self.config["num_layers"],
                    bidirectional=self.config["bidirectional"],batch_first=True)
        elif self.config["rnn_type"] == "gru":
            self.rnn = nn.GRU(self.config["embedding_dim"],self.config["hidden_dim"],num_layers=self.config["num_layers"],
                    bidirectional=self.config["bidirectional"],batch_first=True)
        else:
            raise ValueError("Unknow model name %s"%self.config["rnn_type"])
        if self.config["bidirectional"]:
            self.lin = nn.Linear(self.config["hidden_dim"]*2*self.config["num_layers"],self.config["n_class"])
        else:
            self.lin = nn.Linear(self.config["hidden_dim"]*self.config["num_layers"],self.config["n_class"])

    def forward(self,in_ids):
        batch_size = in_ids.shape[0]
        in_embed = self.word_embeddings(in_ids)
        if self.config["rnn_type"]=="lstm":
            _,(hid_tensor,_) = self.rnn(in_embed)
        elif self.config["rnn_type"] == "gru":
            _,hid_tensor = self.rnn(in_embed)
        hid_tensor = hid_tensor.reshape(batch_size,-1)
        out_put = self.lin(hid_tensor)
        return F.log_softmax(out_put,dim=1)
class RCNN(nn.Module):
    """
    Recurrent Convolutional Neural Networks for Text Classification (2015)
    """
    def __init__(self, n_class,**kwargs):
        super(RCNN, self).__init__()
        self.config = {
            "n_class":n_class,
            "vocab_size":5000,
            "embedding_dim":300,
            "hidden_dim":100,
            "linear_dim":50,
            "num_layers":1,
            "bidirectional":True,
            "rnn_type":"lstm"
        }
        for key in kwargs:
            if key in self.config:
                self.config[key] = kwargs[key]
        self.word_embeddings = nn.Embedding(self.config["vocab_size"],self.config["embedding_dim"], padding_idx=0)
        if self.config["rnn_type"]=="lstm":
            self.rnn = nn.LSTM(self.config["embedding_dim"],self.config["hidden_dim"],num_layers=self.config["num_layers"],
                    bidirectional=self.config["bidirectional"],batch_first=True)
        elif self.config["rnn_type"] == "gru":
            self.rnn = nn.GRU(self.config["embedding_dim"],self.config["hidden_dim"],num_layers=self.config["num_layers"],
                    bidirectional=self.config["bidirectional"],batch_first=True)
        else:
            raise ValueError("Unknow model name %s"%self.config["rnn_type"])
        self.W = nn.Linear(self.config["embedding_dim"] + 2*self.config["hidden_dim"], self.config["linear_dim"])
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(self.config["linear_dim"], n_class)

    def forward(self, in_ids):
        x_emb = self.word_embeddings(in_ids) # (batch_size, seq_len, embedding_dim)
        if self.config["rnn_type"]=="lstm":
            output,_ = self.rnn(x_emb)
        elif self.config["rnn_type"] == "gru":
            output,_ = self.rnn(x_emb) # (batch_size, seq_len, 2*hidden_dim)
        else:
            raise ValueError("Unknow model name %s"%self.config["rnn_type"])
        output = torch.cat([output, x_emb], 2) # (batch_size, seq_len, embedding_dim + 2*hidden_dim)
        output = self.tanh(self.W(output)) # (batch_size, seq_len, linear_dim)
        output = output.transpose(1, 2) # (batch_size,linear_dim,seq_len)
        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        output = self.fc(output)
        return F.log_softmax(output,dim=1)
def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)
class CapsAttNet(nn.Module):
    def __init__(self,n_class,**kwargs):
        """
        Initialize the layer.

        Args:
            in_dim: 		Dimensionality (i.e. length) of each capsule vector.
            num_caps: 		Number of capsules in the capsule layer
            dim_caps: 		Dimensionality, i.e. length, of the output capsule vector.
            num_routing:	Number of iterations during routing algorithm
        """
        super(CapsAttNet,self).__init__()
        self.config = {
            "n_class":n_class,
            "vocab_size":5000,
            "embedding_dim":300,
            "num_capsules":4,
            "capsules_dim":100,
            "hidden_dim":50,
            "num_routing":3
        }
        for key in kwargs:
            if key in self.config:
                self.config[key] = kwargs[key]
        self.word_embeddings = nn.Embedding(self.config["vocab_size"],self.config["embedding_dim"], padding_idx=0)
        self.capsule_layer = CapsuleLayer(self.config["num_capsules"],self.config["embedding_dim"],
                            self.config["capsules_dim"],self.config["num_routing"])
        self.att = SelfAttLayer(self.config["capsules_dim"],self.config["hidden_dim"])
        self.fc = nn.Linear(self.config["capsules_dim"],self.config["n_class"])
    def forward(self,in_ids):
        x_embed = self.word_embeddings(in_ids) # (batch_size,seq_len,embedding_dim)
        output = self.capsule_layer(x_embed) # (batch_size,capsules_dim)
        output = self.att(output)
        output = self.fc(output)
        return F.log_softmax(output,dim=1)
class DPCNN(nn.Module):
    def __init__(self,n_class,**kwargs):
        super(DPCNN,self).__init__()
        self.config = {
            "n_class":n_class,
            "vocab_size":5000,
            "embedding_dim":300,
            "kernerl_word":3,
            "num_filters":8
        }
        for key in kwargs:
            if key in self.config:
                self.config[key] = kwargs[key]
        self.word_embeddings = nn.Embedding(self.config["vocab_size"],self.config["embedding_dim"], padding_idx=0)
        self.conv_region = nn.Conv2d(1,self.config["num_filters"], (3,self.config["embedding_dim"]), stride=1)
        self.conv = nn.Conv2d(self.config["num_filters"],self.config["num_filters"], (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.config["num_filters"], self.config["n_class"])
    def forward(self,in_ids):
        # Embedding Layer
        x_embed = self.word_embeddings(in_ids)
        x_embed = x_embed.unsqueeze(1)# (batch_size,1,embedding_dim, seq_len)
        # First convolutional layer
        x = self.conv_region(x_embed)  # (batch_size, embedding_dim, seq_len-3+1, 1)
        x = self.padding1(x)  # (batch_size, embedding_dim, seq_len, 1)
        x = self.relu(x)
        # Second convolutional layer
        x = self.conv(x)  # (batch_size, embedding_dim, seq_len-3+1, 1)
        x = self.padding1(x)  # (batch_size, embedding_dim, seq_len, 1)
        x = self.relu(x)
        # Thrid convolutional layer
        x = self.conv(x)  # (batch_size, embedding_dim, seq_len-3+1, 1)
        while x.shape[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # (batch_size, num_filters)
        x = self.fc(x)
        return F.log_softmax(x,dim=1)
    def _block(self, x):
        # first convolution
        x = self.padding2(x)
        px = self.max_pool(x)
        print("px",px.shape)
        x = self.padding1(px)
        x = F.relu(x)
        # second convolution
        x = self.conv(x)
        x = self.padding1(x)
        x = F.relu(x)
        # third convolution
        x = self.conv(x)
        # res 
        x = x + px
        return x