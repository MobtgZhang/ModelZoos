import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def squash(x, dim=-1):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm)
    return scale * x / (squared_norm.sqrt() + 1e-8)
class CapsuleLayer(nn.Module):
    def __init__(self,num_capsules,in_dim,capsules_dim,num_routing):
        super(CapsuleLayer,self).__init__()
        self.num_capsules = num_capsules
        self.in_dim = in_dim
        self.capsules_dim = capsules_dim
        self.num_routing = num_routing

        self.W = nn.Parameter(0.001*torch.randn(num_capsules,in_dim,capsules_dim),
                              requires_grad=True)
    def forward(self,x_tensor):
        """
        x_tensor: (batch_size,seq_len,embedding_dim)
        """
        batch_size = x_tensor.size(0)
        device = x_tensor.device
        x_tensor = x_tensor.unsqueeze(1) # (batch_size,1,seq_len,embedding_dim)
        u_hat = torch.matmul(x_tensor,self.W) 
        # W @ x = (batch_size, 1, seq_len, embedding_dim) @ (num_capsules,embedding_dim,capsules_dim) =
        # (batch_size, num_capsules, seq_len, capsules_dim)
        # detach u_hat during routing iterations to prevent gradients from flowing
        temp_u_hat = u_hat.detach()
        in_caps = temp_u_hat.shape[2]
        b = torch.rand(batch_size, self.num_capsules, in_caps, 1).to(device)
        for route_iter in range(self.num_routing - 1):
            c = b.softmax(dim=1)
            # element-wise multiplication
            c_extend = c.expand_as(temp_u_hat)
            s = (c_extend * temp_u_hat).sum(dim=2)
            v = squash(s)
            # dot product agreement between the current output vj and the prediction uj|i
            # (batch_size, num_capsules, seq_len, capsules_dim) @ (batch_size, num_capsules, capsules_dim, 1)
            # -> (batch_size, num_capsules, seq_len, 1)
            uv = torch.matmul(temp_u_hat, v.unsqueeze(-1))
            b += uv
        # last iteration is done on the original u_hat, without the routing weights update
        c = b.softmax(dim=1)
        c_extend = c.expand_as(u_hat)
        s = (c * u_hat).sum(dim=2)
        # apply "squashing" non-linearity along dim_caps
        v = squash(s)
        return v
class SelfAttLayer(nn.Module):
    def __init__(self,in_dim,hid_dim):
        super(SelfAttLayer,self).__init__()
        self.lin = nn.Linear(in_dim,hid_dim)
    def forward(self,input_tensor):
        """
        input tensor:(batch_size,seq_len,in_dim)
        output tensor:(batch_size,in_dim)
        """
        u_tensor = torch.tanh(self.lin(input_tensor)) # (batch_size,seq_len,hid_dim)
        a_tensor = F.softmax(u_tensor,dim=1)
        o_tensor = torch.bmm(a_tensor.transpose(2,1),input_tensor) # (b,h,s)*(b,s,i)->(b,h,i)
        return o_tensor.sum(dim=1)
class MutiSelfAttLayer(nn.Module):
    def __init__(self,in_size,dim_k,dim_v):
        super(MutiSelfAttLayer, self).__init__()
        self.in_size = in_size
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.q_mat = nn.Linear(in_size,dim_k)
        self.k_mat = nn.Linear(in_size,dim_k)
        self.v_mat = nn.Linear(in_size,dim_v)
        self._norm_fact = 1.0/math.sqrt(dim_k)

    def forward(self,hidden_state):
        q_mat = self.q_mat(hidden_state) # (batch_size,seq_len,dim_k)
        k_mat = self.k_mat(hidden_state) # (batch_size,seq_len,dim_k)
        v_mat = self.v_mat(hidden_state) # (batch_size,seq_len,dim_v)
        atten = torch.bmm(q_mat,k_mat.permute(0,2,1))*self._norm_fact # (batch_size,seq_len,seq_len)
        atten = F.softmax(atten,dim=-1)
        o_mat = torch.bmm(atten,v_mat) # (batch_size,seq_len,dim_v)
        o_mat = o_mat.sum(dim=1)



