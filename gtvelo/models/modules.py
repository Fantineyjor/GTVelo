
import torch
import torch.nn as nn
import torch.nn.functional as F
class DynamicDimensionAdjuster(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DynamicDimensionAdjuster, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim).cuda()

    def forward(self, x):
        return self.linear(x)
execution_count = 0
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将 scipy 的稀疏矩阵转换为 PyTorch 的稀疏张量."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)
class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, nhead):
        super(GraphConvLayer, self).__init__()
        if in_features % nhead != 0:
            in_features = (in_features // nhead) * nhead
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features)).to(device)
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.FloatTensor(out_features)).to(device)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        import scipy.sparse as sp
        if isinstance(adj, sp.csr_matrix):
            adj = sparse_mx_to_torch_sparse_tensor(adj).cuda()
        degree = torch.sparse.sum(adj, dim=1).to_dense()
        
        degree[degree == 0] = 1e-5 

        D_inv_sqrt = torch.pow(degree+1e-5, -0.5)
  
        D_inv_sqrt = torch.diag(D_inv_sqrt).to_sparse()

        adj_normalized = torch.sparse.mm(torch.sparse.mm(D_inv_sqrt, adj), D_inv_sqrt)
        out = torch.sparse.mm(adj_normalized, x)
        out = torch.matmul(out, self.weight) + self.bias

        nan_count_out1 = torch.isnan(x).sum().item()
        if nan_count_out1 > 0:
            print(f"x contains {nan_count_out1} NaN values.")
        nan_count_out = torch.isnan(out).sum().item()
        if nan_count_out > 0:
            print(f"GraphConvLayer output contains {nan_count_out} NaN values.")

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F

def adjust_nhead(d_model, nhead):
    """
    调整 nhead 使其能整除 d_model，并选择离原始 nhead 最近的值。
    """
    possible_nheads = [i for i in range(1, d_model + 1) if d_model % i == 0]
    closest_nhead = min(possible_nheads, key=lambda x: abs(x - nhead))
    
    return closest_nhead

class GraphormerLayer(nn.Module):
    def __init__(self, d_model, nhead, out_dim, dim_feedforward=2048, dropout=0.1):
        super(GraphormerLayer, self).__init__()
        self.d_model = d_model
        self.out_dim = out_dim
        self.nhead = adjust_nhead(d_model, nhead)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.self_attn = nn.MultiheadAttention(self.d_model, self.nhead, dropout=dropout).to(device)
        self.linear1 = nn.Linear(self.d_model, dim_feedforward).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.linear2 = nn.Linear(dim_feedforward, self.d_model).to(device)
        self.norm1 = nn.LayerNorm(self.d_model).to(device)
        self.norm2 = nn.LayerNorm(self.d_model).to(device)
        self.dropout1 = nn.Dropout(dropout).to(device)
        self.dropout2 = nn.Dropout(dropout).to(device)
        self.graph_positional_encoding = nn.Parameter(torch.randn(self.d_model))* 1e-3
        self.graph_conv = GraphConvLayer(self.d_model, self.d_model, self.nhead)
        self.positional_linear = None
        self.output_layer = nn.Linear(self.d_model, self.out_dim).to(device)

    def forward(self, x, adj):
        # 获取 x 的维度
        
        batch_size, feature_dim = x.shape
        nan_count4 = torch.isnan(x).sum().item()
        if feature_dim != self.d_model:
            x = F.pad(x, (0, self.d_model - feature_dim))

            feature_dim = self.d_model
        if self.positional_linear is None:
            self.positional_linear = nn.Linear(self.graph_positional_encoding.shape[0], feature_dim).to(x.device)
        positional_encoding_expanded = self.positional_linear(self.graph_positional_encoding.cuda())
 
        x = x + positional_encoding_expanded.unsqueeze(0)
 
        x = x.unsqueeze(0) 
     
        x = F.normalize(x, p=2, dim=-1)


        attn_output, _ = self.self_attn(x, x, x) 
        attn_output=attn_output+ 1e-6  # 自注意力输出
        attn_output = torch.clamp(attn_output, min=-1e5, max=1e5)  # 或更适合的阈值

        x = x.squeeze(0)  # 去掉 seq_len 维度
        # 第一层残差连接和归一化
        x2 = x + self.dropout1(attn_output.squeeze(0))

        x = self.norm1(x2)

        # 图结构编码
        out = self.graph_conv(x, adj)
        x2 = self.linear2(self.dropout(F.relu(self.linear1(out))))
        x = out + self.dropout2(x2)
        x = self.norm2(x)
        x = self.output_layer(x)
        return x 

import numpy as np

def adjust_nhead(d_model, nhead):
    """
    调整 nhead 使其能整除 d_model，并选择离原始 nhead 最近的值。
    """

    possible_nheads = [i for i in range(1, d_model + 1) if d_model % i == 0]
    closest_nhead = min(possible_nheads, key=lambda x: abs(x - nhead))
    
    return closest_nhead



class TransImg(nn.Module):
    def __init__(self, in_dim,  out_dim, dim_feedforward=2048, dropout=0.1,nhead=8):
        super(TransImg, self).__init__()

        # 图 Transformer 层
        self.transformer = GraphormerLayer(in_dim,nhead,out_dim,  dim_feedforward, dropout)

    def forward(self, x, adj):
        y = self.transformer(x, adj)

        return y


def create_encoder(observed, latent, encoder_hidden,nhead=8 ):
    # 创建输入特征的处理层
    encoder_z0_s = TransImg(in_dim=observed,nhead=nhead, out_dim= latent, dim_feedforward=encoder_hidden, dropout=0.1)
    encoder_z0_u = TransImg(in_dim=observed, nhead=nhead, out_dim= latent,dim_feedforward=encoder_hidden, dropout=0.1)

    encoder_z_s = encoder_z0_s
    encoder_z_u = encoder_z0_u

    return encoder_z0_s, encoder_z0_u, encoder_z_s, encoder_z_u

def create_decoder(latent,  observed, decoder_hidden,nhead=8):
    decoder_s = TransImg(in_dim=latent,nhead=nhead,out_dim=observed,  dim_feedforward=decoder_hidden, dropout=0.1)
    decoder_u = TransImg(in_dim=latent,nhead=nhead, out_dim=observed, dim_feedforward=decoder_hidden, dropout=0.1)

    return decoder_s, decoder_u

import torch
import torch.nn as nn
class MLP(nn.Module):
    """
    Simple dense neural network with an additional layer to adjust input dimensions.
    """
    def __init__(self, input, hidden, output, activation=nn.ELU(), bn=False):
        super(MLP, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.activation = activation.to(device)
        self.input = input 
        self.output = output 
        self.hidden = hidden 
        self.layers = nn.ModuleList().to(device)
        self.adjust_layer = None
        self.layers.append(nn.Linear(input, hidden[0])).to(device)
        if bn:
            self.layers.append(nn.BatchNorm1d(hidden[0]))
        self.layers.append(activation).to(device)
        for in_size, out_size in zip(hidden[:-1], hidden[1:]):
            self.layers.append(nn.Linear(in_size, out_size))
            if bn:
                self.layers.append(nn.BatchNorm1d(out_size))
            self.layers.append(activation)
        self.layers.append(nn.Linear(hidden[-1], output)).to(device)

    def forward(self, z):
        if self.adjust_layer is None:
            self.adjust_layer = nn.Linear(z.shape[-1], self.input).to(z.device)
        z = self.adjust_layer(z)
        output = self.layers[0](z) 
        for layer in self.layers[1:]:
            output = layer(output)
        return output


