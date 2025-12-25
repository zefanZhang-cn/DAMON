from transformers.activations import gelu
import torch.nn as nn
import numpy as np
import torch
import math
import copy
from transformers.modeling_outputs import BaseModelOutput
from transformers import BertConfig
from transformers import RobertaConfig
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, skip=True):
        super(GraphConvolution, self).__init__()
        self.skip = skip
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # TODO make fc more efficient via "pack_padded_sequence"

        support = torch.bmm(input, self.weight.unsqueeze(
            0).expand(input.shape[0], -1, -1))
        output = torch.bmm(adj, support)
        # output = SparseMM(adj)(support)
        if self.bias is not None:
            output += self.bias.unsqueeze(0).expand(input.shape[0], -1, -1)
        if self.skip:
            output += support

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Graph(nn.Module):

    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, dropout):
        super(Graph, self).__init__()
        self.fc_k = nn.Linear(dim_in, dim_hidden)
        self.fc_q = nn.Linear(dim_in, dim_hidden)

        dim_hidden = dim_out if num_layers == 1 else dim_hidden
        self.layers = nn.ModuleList([
            GraphConvolution(dim_in, dim_hidden)
        ])

        for i in range(num_layers - 1):
            dim_tmp = dim_out if i == num_layers - 2 else dim_hidden
            self.layers.append(GraphConvolution(dim_hidden, dim_tmp))

        self.dropout = dropout

    def build_graph(self, x):
        batch_size, s_len = x.shape[0], x.shape[1]
        emb_k = self.fc_k(x)
        emb_q = self.fc_q(x)
        length = torch.tensor([s_len] * batch_size, dtype=torch.long)

        s = torch.bmm(emb_k, emb_q.transpose(1, 2))

        s_mask = s.data.new(*s.size()).fill_(1).bool()  # [B, T1, T2]
        # Init similarity mask using lengths
        for i, (l_1, l_2) in enumerate(zip(length, length)):
            s_mask[i][:l_1, :l_2] = 0
        s_mask = Variable(s_mask)
        s.data.masked_fill_(s_mask.data, -float("inf"))

        A = s  # F.softmax(s, dim=2)  # [B, t1, t2]

        # remove nan from softmax on -inf
        A.data.masked_fill_(A.data != A.data, 0)

        return A

    def forward(self, X, A):
        for layer in self.layers:
            X = F.relu(layer(X, A))
            X = F.dropout(X, self.dropout, training=self.training)
        return X
def get_mask(lengths, max_length):
    """ Computes a batch of padding masks given batched lengths """
    mask = 1 * (
        torch.arange(max_length).unsqueeze(1).to(lengths.device) < lengths
    ).transpose(0, 1)
    return mask

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, op = 'n'):
        super().__init__()

        ################BERT & RoBERTa#################
        if 'n' in op:
            self.n_heads = 8  # config.n_heads
            self.dim = 512  # config.dim
        else:
            self.n_heads = 1  # config.n_heads
            self.dim = 256*256


        dp_rate = 0.2  # config.attention_dropout

        ################DisVGT#################
        # self.n_heads = config.n_heads
        # self.dim = config.dim
        # dp_rate = config.attention_dropout

        self.dropout = nn.Dropout(p=dp_rate)

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.k_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.v_lin = nn.Linear(in_features=self.dim, out_features=self.dim)
        self.out_lin = nn.Linear(in_features=self.dim, out_features=self.dim)

        self.pruned_heads = set()

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)
            )
        print("query::::",query.shape)
        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        # mask = (
        #     (mask == 0).view(mask_reshp).expand_as(scores)
        # )  # (bs, n_heads, q_length, k_length)
        # scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class FFN(nn.Module):
    def __init__(self,op = 'n'):
        super().__init__()
        if 'n' in op:
            dropout, dim, hidden_dim = 0.2, 512, 512
        else:
            dropout, dim, hidden_dim = 0.1, 256*256, 256*256
        activation = 'gelu'
        ##########DisVGT###############
        # dropout, dim, hidden_dim = config.attention_dropout, config.dim, config.hidden_dim
        # activation = config.activation

        self.dropout = nn.Dropout(p=dropout)
        self.lin1 = nn.Linear(in_features=dim, out_features=hidden_dim)
        self.lin2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        assert activation in [
            "relu",
            "gelu",
        ], "activation ({}) must be in ['relu', 'gelu']".format(activation)
        self.activation = gelu if activation == "gelu" else nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, op = 'n'):
        super().__init__()
        if 'n' in op:
            dim = 512
        else :
            dim = 256*256
        #assert config.hidden_size % config.num_attention_heads == 0
        #########DisVGT##########
        # dim = config.dim
        # assert config.dim % config.n_heads == 0

        self.attention = MultiHeadSelfAttention(op)
        self.sa_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

        self.ffn = FFN(op)
        self.output_layer_norm = nn.LayerNorm(normalized_shape=dim, eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            (
                sa_output,
                sa_weights,
            ) = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attention` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(
            ffn_output + sa_output
        )  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output


class Transformer(nn.Module):
    def __init__(self,op = 'n'):
        super().__init__()
        self.n_layers = 2
        # if 'n' in op:
        #     self.n_layers = 2
        # else:
        #     self.n_layers = 4
        ############DisBERT################
        # self.n_layers = config.n_layers

        layer = TransformerBlock(op)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(self.n_layers)]
        )

    def forward(
            self,
            x,
            attn_mask=None,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=None,
    ):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
            if head_mask is not None:
                layer_outputs = layer_module(
                    x=hidden_state,
                    attn_mask=attn_mask,
                    head_mask=head_mask[i],
                    output_attentions=output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    x=hidden_state,
                    attn_mask=attn_mask,
                    head_mask=None,
                    output_attentions=output_attentions,
                )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_state, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
