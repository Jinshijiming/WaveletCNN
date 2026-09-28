"""
"Abandon Locality: Frame-Wise Embedding Aided Transformer for Automatic Modulation Recognition"
https://ieeexplore.ieee.org/document/9915584

reproduced by ASME
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from math import sqrt
from .. import BaseModel

class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.contiguous().view(B, L, -1)

        return self.out_projection(out), attn


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., activation="relu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class DB_FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0., activation="relu"):
        super().__init__()
        self.projection1 = nn.Linear(d_model, d_ff)
        self.projection2 = nn.Linear(d_model, d_ff)
        self.projection = nn.Linear(d_ff, d_model)
        # self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.projection1(x)
        x2 = self.projection2(x)
        y1 = F.gelu(x1) * x2
        y2 = F.gelu(x2) * x1
        return self.projection(y1 + y2)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.FFlayer = DB_FeedForward(d_model, d_ff, dropout=dropout, activation=activation)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = self.FFlayer(x)
        return self.norm(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Net(BaseModel):
    def __init__(self, L, R, length, num_classes, d_model, d_ff=None, e_layers=2, n_heads=1,
                 dropout=0.5, activation='relu', use_cls=False, use_pos=True):
        super().__init__()
        self.L = L
        self.R = R
        self.F = (length - L) // R + 1

        self.use_cls = use_cls
        self.use_pos = use_pos

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) if self.use_cls else None
        self.pos_embedding = nn.Parameter(torch.randn(1, self.F + 1, d_model))
        self.enc_embedding = nn.Sequential(
            Rearrange('b l c -> b c l'),
            nn.BatchNorm1d(2),
            # Rearrange('b l c-> b 1 l c')
            Rearrange('b c l-> b 1 l c'),
            nn.Unfold(kernel_size=(L, 2), stride=(R, 2)),
            Rearrange('b L F-> b F L'),
            nn.Linear(2 * L, d_model)
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(FullAttention(), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout,
                    activation
                ) for _ in range(e_layers)
            ],
            # norm_layer=nn.LayerNorm(d_model)
        )


        self.classfier = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Linear(2 * d_model, num_classes)
        )

    def forward(self, data):
        """

        :param data: [B,L,2]
        :return:
        """
        # Norm

        # embedding
        enc_out = self.enc_embedding(data)
        B, L, _ = enc_out.shape
        # If we want to add a class token, add it here.
        if self.use_cls:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
            enc_out = torch.cat((cls_tokens, enc_out), dim=1)
            enc_out += self.pos_embedding[:, :(L + 1)] if self.use_pos else enc_out
        else:
            enc_out += self.pos_embedding[:, :L] if self.use_pos else enc_out
        # encoder
        x, attns = self.encoder(enc_out)
        x = x[:, 0] if self.use_cls else x.mean(dim=1)

        x = self.classfier(x)
        return x



if __name__ == '__main__':
    model = Net(L=16, R=4, length=128, num_classes=7, d_model=64, e_layers=2, n_heads=4, dropout=0.1, activation='gelu', use_cls=True, use_pos=True)
    x = torch.randn(2, 128, 2)
    y = model(x)
    print(y.shape)