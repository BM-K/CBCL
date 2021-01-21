import torch
import torch.nn as nn
import math
from transformers import BertModel, BertConfig
import numpy as np

import logging
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


def get_attn_pad_mask(args, seq_q, seq_k, pad_idx):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(pad_idx).unsqueeze(1)  # BERT PAD = 0 이므로 eq(0)
    # batch_size x 1 x len_k(=len_q), one is masking

    return pad_attn_mask.expand(batch_size, len_q, len_k).to(args.device)  # batch_size x len_q x len_k


def get_attn_subsequent_mask(args, seq):
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)  # 상삼각행렬 반환
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask.to(args.device)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, args):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = int(args.d_model / args.n_heads)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)
        scores.masked_fill_(attn_mask, -1e9)
        # padding 부분을 -1000000 처럼 큰 음수값을 할당하여 softmax 후 해당 값을 0으로 나오게 만들어야함.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiheadAttention(nn.Module):
    def __init__(self, args):
        super(MultiheadAttention, self).__init__()
        self.args = args
        self.d_k = int(args.d_model / args.n_heads)
        self.d_v = int(args.d_model / args.n_heads)
        self.n_heads = args.n_heads
        self.W_Q = nn.Linear(args.d_model, self.d_k * args.n_heads)  # init (512 x 64 * 8)
        self.W_K = nn.Linear(args.d_model, self.d_k * args.n_heads)
        self.W_V = nn.Linear(args.d_model, self.d_v * args.n_heads)
        self.li1 = nn.Linear(args.n_heads * self.d_v, args.d_model)
        self.layer_norm = nn.LayerNorm(args.d_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch x len_k x d_model]
        residual, batch_size = Q, Q.size(0)

        # print(Q.size(), K.size(), V.size())
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                 2)  # q_s:[batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                 2)  # k_s:[batch_size x n_heads x len_q x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                                 2)  # v_s:[batch_size x n_heads x len_q x d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.n_heads, 1, 1)

        context, attn = ScaledDotProductAttention(self.args)(q_s, k_s, v_s, attn_mask)

        # print(q_s.size(), k_s.size(), v_s.size())

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.li1(context)

        return self.layer_norm(output + residual), attn
        # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=args.d_model, out_channels=args.feedforward, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=args.feedforward, out_channels=args.d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(args.d_model)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = self.relu(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


class DecoderLayer(nn.Module):
    def __init__(self, args):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiheadAttention(args)
        self.dec_enc_attn = MultiheadAttention(args)
        self.pos_ffn = PoswiseFeedForwardNet(args)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs


class Decoder(nn.Module):
    def __init__(self, args, vocab_size):
        super(Decoder, self).__init__()
        self.args = args
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])
        self.vocab_embedding = nn.Embedding(vocab_size, args.d_model)
        self.pos_embedding = PositionalEncoding(args.d_model, args.max_len)

    def forward(self, enc_inputs, dec_inputs, enc_outputs, data_loader):  # dec_inputs : [batch_size x target_len]
        dec_outputs = self.vocab_embedding(dec_inputs)
        dec_outputs = dec_outputs + self.pos_embedding(dec_inputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(self.args, dec_inputs, dec_inputs, data_loader.pad_token_idx)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(self.args, dec_inputs)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(self.args, dec_inputs, enc_inputs, data_loader.pad_token_idx)

        for layer in self.layers:
            dec_outputs = layer(
                dec_outputs,
                enc_outputs,
                dec_self_attn_mask,
                dec_enc_attn_mask,)

        return dec_outputs


class ETRI_KOBERT(nn.Module):
    def __init__(self, args):
        super(ETRI_KOBERT, self).__init__()
        self.args = args
        # self.li1 = nn.Linear(768, args.embedding_dim)
        self.model = BertModel.from_pretrained("./ETRI_KoBERT/003_bert_eojeol_pytorch")
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, opt):
        logits, _ = self.model(input_ids=opt['inputs'],
                               token_type_ids=opt['segment_ids'],
                               attention_mask=opt['attention_mask'])
        # top_vec = self.li1(top_vec)
        return logits.to(self.args.device)#self.dropout(logits.to(self.args.device))


class Transformer(nn.Module):
    def __init__(self, args, opt):
        super(Transformer, self).__init__()
        self.special_tokens_length = len(opt.special_tokens_dict)
        self.bert = ETRI_KOBERT(args)

        self.vocab_size = self.bert.model.config.vocab_size + self.special_tokens_length
        self.bert_config = BertConfig(self.vocab_size,
                                      hidden_size=args.d_model,
                                      num_hidden_layers=args.n_heads,
                                      num_attention_heads=args.n_layers)
        self.bert.model = BertModel(self.bert_config)
        logger.info(self.bert.model.config)

        self.args = args
        self.dropout = args.dropout

        self.decoder = Decoder(args, self.vocab_size)
        self.projection = nn.Linear(args.d_model, self.vocab_size, bias=False)

    def forward(self, bert_opt, de_input, obj):
        bert_logits = self.bert(bert_opt)
        dec_outputs = self.decoder(bert_opt['inputs'], de_input, bert_logits, obj['dataloader'])
        dec_logits = self.projection(dec_outputs)

        return dec_logits.view(-1, dec_logits.size(-1))