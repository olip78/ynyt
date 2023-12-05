# a very slight modification of https://github.com/AzadDeihim/STTRE/blob/main/STTRE.ipynb
# article: https://www.sciencedirect.com/science/article/pii/S0893608023005361?ssrnid=4404879&dgcid=SSRN_redirect_SD

from torch.nn import functional as F
from torch import nn
import torch


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, rel_emb, device, mode='encoder'):
        super(SelfAttention, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.heads = heads
        self.seq_len = seq_len
        self.module = module
        self.rel_emb = rel_emb
        self.mode = mode

        if module == 'spatial' or module == 'temporal':
            self.head_dim = seq_len
            self.values = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32)
            self.keys = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32, device=self.device)
            self.queries = nn.Linear(self.embed_size, self.embed_size, dtype=torch.float32, device=self.device)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([self.heads, self.head_dim, self.embed_size], device=self.device))

        else:
            self.head_dim = embed_size // heads
            assert (self.head_dim * heads == embed_size), "Embed size not div by heads"
            self.values = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32)
            self.keys = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32, device=self.device)
            self.queries = nn.Linear(self.head_dim, self.head_dim, dtype=torch.float32, device=self.device)

            if rel_emb:
                self.E = nn.Parameter(torch.randn([1, self.seq_len, self.head_dim], device=self.device))

        self.fc_out = nn.Linear(self.embed_size, self.embed_size, device=self.device)

        self.module = module

    def forward(self, x):
        N, _, _, _  = x.shape
        x = x[:, :, :, 0]

        # non-shared weights between heads for spatial and temporal modules
        if self.module == 'spatial' or self.module == 'temporal':
            values = self.values(x)
            keys = self.keys(x)
            queries = self.queries(x)
            values = values.reshape(N, self.seq_len, self.heads, self.embed_size)
            keys = keys.reshape(N, self.seq_len, self.heads, self.embed_size)
            queries = queries.reshape(N, self.seq_len, self.heads, self.embed_size)
        else:
            z = x.reshape(N, self.seq_len, self.heads, self.head_dim)
            values = self.values(z)
            keys = self.keys(z)
            queries = self.queries(z)

        if self.mode == 'decoder':
            mask = torch.triu(torch.ones(1, self.seq_len, self.seq_len, device=self.device), 1)
        else:
            mask = None

        if self.rel_emb:
            QE = torch.matmul(queries.transpose(1, 2), self.E.transpose(1, 2))
            QE = self._mask_positions(QE)
            S = self._skew(QE).contiguous().view(N, self.heads, self.seq_len, self.seq_len)
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            
            if mask is not None:
                qk = qk.masked_fill(mask == 0, float("-1e20"))

            attention = torch.softmax(qk / (self.embed_size ** (1 / 2)), dim=3) + S
        else:
            qk = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
            
            if mask is not None:
                qk = qk.masked_fill(mask == 0, float("-1e20"))

            attention = torch.softmax(qk / (self.embed_size ** (1 / 2)), dim=3)

        # attention(N x Heads x Q_Len x K_len)
        # values(N x V_Len x Heads x Head_dim)
        # z(N x Q_Len x Heads * Head_dim)

        if self.module == 'spatial' or self.module == 'temporal':
            z = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, self.seq_len * self.heads, self.embed_size)
        else:
            z = torch.einsum("nhql,nlhd->nqhd", [attention, values])
            z = z.reshape(N, self.seq_len, self.heads * self.head_dim)

        z = self.fc_out(z)

        return z


    def _mask_positions(self, qe):
        L = qe.shape[-1]
        mask = torch.triu(torch.ones(L, L, device=self.device), 1).flip(1)
        return qe.masked_fill((mask == 1), 0)

    def _skew(self, qe):
        #pad a column of zeros on the left
        padded_qe = F.pad(qe, [1,0])
        s = padded_qe.shape
        padded_qe = padded_qe.view(s[0], s[1], s[3], s[2])
        #take out first (padded) row
        return padded_qe[:,:,1:,:]


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, forward_expansion, rel_emb, device):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, seq_len, module, rel_emb=rel_emb, device=device)

        if module == 'spatial' or module == 'temporal':
            self.norm1 = nn.BatchNorm1d(seq_len*heads)
            self.norm2 = nn.BatchNorm1d(seq_len*heads)
        else:
            self.norm1 = nn.BatchNorm1d(seq_len)
            self.norm2 = nn.BatchNorm1d(seq_len)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.LeakyReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.module = module

    def forward(self, x):
        z = x[:, :, :, 0]
        attention = self.attention(x)
        x = self.norm1(attention + z)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out


class Encoder(nn.Module):
    def __init__(self, seq_len, embed_size, num_layers, heads, device,
                 forward_expansion, module,
                 rel_emb=True):
        super(Encoder, self).__init__()
        self.module = module
        self.embed_size = embed_size
        self.device = device
        self.rel_emb = rel_emb
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.layers = nn.ModuleList(
            [
             TransformerBlock(embed_size, heads, seq_len, module, forward_expansion=forward_expansion, 
                   rel_emb = rel_emb,  device=device)
             for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
        out = self.fc_out(out)
        return out
