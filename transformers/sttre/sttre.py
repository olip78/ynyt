# a modification of https://github.com/AzadDeihim/STTRE/blob/main/STTRE.ipynb
# article: https://www.sciencedirect.com/science/article/pii/S0893608023005361?ssrnid=4404879&dgcid=SSRN_redirect_SD

from torch.nn import functional as F
from torch import nn
import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsolutePercentageError, MeanAbsoluteError, R2Score


from tqdm import tqdm
from functools import partialmethod

from .data import YNYT


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

        # Xavier initialization
        nn.init.xavier_uniform_(self.values.weight)
        nn.init.xavier_uniform_(self.keys.weight)
        nn.init.xavier_uniform_(self.queries.weight)
        nn.init.xavier_uniform_(self.fc_out.weight)

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

# ------------ Encoder ----------

class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, seq_len, module, forward_expansion, rel_emb, device):
        super(EncoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, seq_len, module, rel_emb=rel_emb, device=device)

        if module == 'spatial' or module == 'temporal':
            self.norm1 = nn.BatchNorm1d(seq_len*heads)
            self.norm2 = nn.BatchNorm1d(seq_len*heads)
            #self.norm1 = nn.LayerNorm(embed_size)
            #self.norm2 = nn.LayerNorm(embed_size)
        else:
            #self.norm1 = nn.BatchNorm1d(seq_len)
            #self.norm2 = nn.BatchNorm1d(seq_len)
            self.norm1 = nn.LayerNorm(embed_size)
            self.norm2 = nn.LayerNorm(embed_size)

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
             EncoderBlock(embed_size, heads, seq_len, module, forward_expansion=forward_expansion, 
                   rel_emb = rel_emb,  device=device)
             for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
        out = self.fc_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, input_shape,
                 embed_size, num_layers, forward_expansion, heads, device, dropout, regression_head, horizon=6):

        super(Transformer, self).__init__()
        self.device = device

        self.batch_size, self.num_features, self.seq_len, self.num_var, _ = input_shape
        self.embed_size = embed_size

        self.element_embedding_temporal = nn.Linear(self.seq_len*self.num_features, embed_size*self.seq_len)
        self.element_embedding_spatial = nn.Linear(self.num_var*self.num_features, embed_size*self.num_var)
        
        self.pos_embedding = nn.Embedding(self.seq_len, embed_size)
        self.variable_embedding = nn.Embedding(self.num_var, embed_size)

        self.temporal = Encoder(seq_len=self.seq_len,
                                embed_size=embed_size,
                                num_layers=num_layers,
                                heads=self.num_var,
                                device=self.device,
                                forward_expansion=forward_expansion,
                                module='temporal',
                                rel_emb=True)

        self.spatial = Encoder(seq_len=self.num_var,
                               embed_size=embed_size,
                               num_layers=num_layers,
                               heads=self.seq_len,
                               device=self.device,
                               forward_expansion=forward_expansion,
                               module = 'spatial',
                               rel_emb=True)

        self.spatiotemporal = Encoder(seq_len=self.seq_len*self.num_var,
                                      embed_size=embed_size,
                                      num_layers=num_layers,
                                      heads=heads,
                                      device=self.device,
                                      forward_expansion=forward_expansion,
                                      module = 'spatiotemporal',
                                      rel_emb=True)
        
        factor = regression_head['flatt_factor']
        
        k = 3
        d_out = (embed_size // factor) * self.seq_len * k * self.num_var
        self.flatter = nn.Sequential(
                                     nn.Linear(embed_size, embed_size // factor),
                                     nn.LeakyReLU(),
                                     nn.Flatten(),
                                     #nn.BatchNorm1d(d_out)
                                    )

        # additional features
        self.add_dim = horizon * 55 * regression_head['add_features']
        d_out += self.add_dim
        
        if regression_head['heads'] == 1:
            output_size = self.num_var * horizon
            self.head = []
            for i, l in enumerate(regression_head['layers']):
                l1 = self.num_var * horizon * l
                if i == 0:
                    self.head.append(nn.Linear(d_out, l1))
                else:
                    l0 = self.num_var * horizon * regression_head['layers'][i - 1]
                    self.head.append(nn.Linear(l0, l1))
                self.head.append(nn.LeakyReLU())
                self.head.append(nn.Dropout(regression_head['dropout_head']))
            self.head.append(nn.Linear(l1, output_size))
            self.head = nn.Sequential(*self.head)
        
    def forward(self, x, regressors, dropout):
        batch_size = len(x)

        #process/embed input for spatio-temporal module
        positions = torch.arange(0, self.seq_len
                                ).expand(batch_size, self.num_var, self.seq_len
                                        ).reshape(batch_size, self.num_var * self.seq_len
                                                 ).to(self.device)

        x_spatio_temporal = x.reshape(batch_size, self.num_var, self.seq_len*self.num_features)
        x_spatio_temporal = self.element_embedding_temporal(x_spatio_temporal
                                    ).reshape(batch_size, self.num_var * self.seq_len, self.embed_size)
        x_spatio_temporal = F.dropout(self.pos_embedding(positions) + x_spatio_temporal, dropout)
        x_spatio_temporal = torch.unsqueeze(x_spatio_temporal, -1)


        #process/embed input for temporal module
        positions = torch.arange(0, self.seq_len
                                ).expand(batch_size, self.num_var, self.seq_len
                                        ).reshape(batch_size, self.num_var * self.seq_len
                                                 ).to(self.device)

        x_temporal = x.view(batch_size, self.num_var, self.seq_len*self.num_features)
        x_temporal = self.element_embedding_temporal(x_temporal
                                    ).reshape(batch_size, self.num_var * self.seq_len, self.embed_size)
        x_temporal = F.dropout(self.pos_embedding(positions) + x_temporal, dropout)
        x_temporal = torch.unsqueeze(x_temporal, -1)
        
        #process/embed input for spatial module
        vars = torch.arange(0, self.num_var).expand(batch_size, self.seq_len, self.num_var).reshape(batch_size, self.num_var*self.seq_len).to(self.device)
        
        x_spatial = x.view(batch_size, self.seq_len, self.num_features*self.num_var)
        x_spatial = self.element_embedding_spatial(x_spatial).reshape(batch_size, self.num_var * self.seq_len, self.embed_size)
        x_spatial = F.dropout(self.variable_embedding(vars) + x_spatial, dropout)
        x_spatial = torch.unsqueeze(x_spatial, -1)
        
        out1 = self.temporal(x_temporal)
        out2 = self.spatial(x_spatial)
        out3 = self.spatiotemporal(x_spatio_temporal)

        out = torch.cat((out1, out2, out3), 1)
        
        out = out.contiguous()


        out = self.flatter(out)

        if self.add_dim > 0:
            out = torch.cat([out.unsqueeze(1), regressors.squeeze(-1)], dim=2).squeeze(1)

        out = self.head(out)

        return out


def train_val(period, epoches, path_preprocessed,
              embed_size, heads, num_layers, dropout, forward_expansion, lr, batch_size, seq_len, 
              regression_head, data_setting, device, verbose=True, verbose_step=1, horizon=6):
    #device = torch.device("mps")
    print(f'{device}!')

    train_data = YNYT(period['train'], seq_len=seq_len, mode='train', path_preprocessed=path_preprocessed, 
                      setting=data_setting, horizon=horizon)
    test_data = YNYT(period['val'], seq_len=seq_len, mode='test', path_preprocessed=path_preprocessed, 
                     setting=data_setting, horizon=horizon)

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    #define loss function, and evaluation metrics
    mape = MeanAbsolutePercentageError().to(device)
    mae = MeanAbsoluteError().to(device)
    r2 = R2Score().to(device)
    loss_fn = torch.nn.MSELoss()

    inputs, _, _ = next(iter(train_dataloader))

    model = Transformer(inputs.shape, embed_size=embed_size, num_layers=num_layers,
                        forward_expansion=forward_expansion, 
                        heads=heads, dropout = dropout,
                        device=device,
                        regression_head=regression_head, horizon=horizon).to(device)
    
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    val_losses = []
    val_mae = []
    val_mape = []

    if verbose:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    else:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)

    for epoch in tqdm(range(epoches)):
        total_loss = 0

        #train loop
        loss_train = 0
        for i, data in enumerate(train_dataloader):
            inputs, regressors, labels = data
            regressors = regressors.unsqueeze(-1)
            labels = torch.flatten(labels, start_dim=1)
            optimizer.zero_grad()
            output = model(inputs.to(device), regressors.to(device), dropout)
            loss = loss_fn(output, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss
        
        loss_train = total_loss / (i + 1)
        
        total_loss = 0
        total_mae = 0
        total_mape = 0
        total_r2 = 0
        div = 1

        #test loop
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                inputs, regressors, labels = data
                regressors = regressors.unsqueeze(-1)
                labels = torch.flatten(labels, start_dim=1)
                output = model(inputs.to(device), regressors.to(device), 0)
                loss = loss_fn(output, labels.to(device))
                total_mae = total_mae + mae(output, labels.to(device))
                total_mape = total_mape + mape(output, labels.to(device))
                total_r2 = total_r2 + r2(torch.flatten(output), torch.flatten(labels.to(device)))
                total_loss = total_loss + loss
                #div is used when the number of samples in a batch is less
                #than the batch size
                div += (len(inputs)/batch_size)

        val_losses.append(total_loss.item()/div)
        val_mae.append(total_mae.item()/div)
        val_mape.append(total_mape.item()/div)

        if verbose and epoch % verbose_step == 0:
            print(f'epoch: {epoch}, train loss: {loss_train.item():.6f}, val loss: {total_loss.item() / (i + 1):.6f}, val mae: {total_mae / (i + 1):.6f}, val r2: {total_r2 / (i + 1):.6f}')
        
    return model