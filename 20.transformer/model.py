import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from d2l import torch as d2l
import matplotlib.pyplot as plt

def sequence_mask(score_mat : torch.Tensor, valid_len : torch.Tensor) -> torch.Tensor:
    assert score_mat.dim() == 3, print(f"score_mat dim is {score_mat.dim()}")
    assert valid_len.dim() == 1, print(f"valid_len dim is {valid_len.dim()}")
    max_len = score_mat.shape[1]
    mask = torch.arange(max_len, device = valid_len.device)[None, :] < valid_len[:, None]
    return mask

def mask_softmax(score : torch.Tensor, valid_len : torch.Tensor) -> torch.Tensor:
    score_mask = sequence_mask(score, valid_len)
    score = score.reshape(-1, score.shape[-1])
    score = score.masked_fill(~score_mask, -1e6)
#     print(score)
    return F.softmax(score, -1)


def postion_embeding(max_len, dim) -> torch.Tensor:
        position_embeding = torch.zeros((max_len, dim))

        position_mat = torch.arange(max_len)[:, None] / torch.pow(10000,
                                                                  torch.arange(0, dim, 2) / dim)[None, :]

        position_embeding[:, 0::2] = torch.sin(position_mat)
        position_embeding[:, 1::2] = torch.cos(position_mat)

        return position_embeding


def show_position(position_embeding: torch.Tensor):
        assert position_embeding.dim() == 2, print(f"enc_input dim is {position_embeding.dim()}")

        num_rows, num_cols = position_embeding.shape

        for col in range(num_cols):
                plt.plot(range(num_rows), position_embeding[:, col])

        plt.title("Postion Embeding")
        plt.xlabel("Seqence idx")
        plt.ylabel("Val")
        plt.xticks(range(num_rows))
        plt.legend()
        plt.grid(True)
        plt.show()


class PositionEmbeding(nn.Module):
        def __init__(self, max_len, dims, dropout):
                super(PositionEmbeding, self).__init__()
                self.pe = postion_embeding(max_len, dims)
                self.drop = nn.Dropout(dropout)

        def forward(self, X):
                assert X.shape[1:] == self.pe.shape, print(f"X shape is not equal to pe {X.shape} vs {self.pe.shape}!")
                return self.drop(X / math.sqrt(X.shape[-1]) + self.pe.to(X.device))

        def show_position(self):
                show_position(self.pe)


class DotProductAttention(nn.Module):
        def __init__(self, dropout):
                super(DotProductAttention, self).__init__()
                self.drop = nn.Dropout(dropout)

        def forward(self, querys: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, valid_lens: torch.Tensor):
                assert querys.dim() == keys.dim() == values.dim() == 3
                self.attention_weights = torch.bmm(querys, keys.transpose(1, 2)) / math.sqrt(querys.shape[-1])
                # [batch_size, max_len, max_len] --> [batch_size * max_len, max_len]
                self.attention_weights = mask_softmax(self.attention_weights, valid_lens)
                self.attention_weights = self.attention_weights.reshape(querys.shape[0], querys.shape[1],
                                                                        keys.shape[1])
                #         print(self.attention_weights)
                return torch.bmm(self.drop(self.attention_weights), values)


class MutilHeadAttention(nn.Module):
        def __init__(self, query_size, key_size, val_size,
                     num_heads, num_hiddens,
                     dropout, bias=False):
                super(MutilHeadAttention, self).__init__()
                assert num_hiddens % num_heads == 0, print(
                        f"num_hiddens:{num_hiddens} or num_heads:{num_heads} set error!")

                self.num_hiddens = num_hiddens
                self.num_heads = num_heads
                self.num_hiddens_one_head = num_hiddens // num_heads
                self.W_Q = nn.Linear(query_size, num_hiddens, bias=bias)
                self.W_K = nn.Linear(key_size, num_hiddens, bias=bias)
                self.W_V = nn.Linear(val_size, num_hiddens, bias=bias)
                self.W_out = nn.Linear(num_hiddens, num_hiddens, bias=bias)

                self.attention = DotProductAttention(dropout)

        def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                    valid_lens: torch.Tensor):
                assert Q.shape[-1] == K.shape[-1] == V.shape[-1] == self.num_hiddens, print(
                        "num_hiddens is not equal to Q, K, V dims")
                assert Q.shape[0] == K.shape[0] == V.shape[0], print("batch size is not equal")
                # assert valid_lens.shape[0] == Q.shape[0] * Q.shape[1] * self.num_heads, print(
                #         f"batch_size:{Q.shape[0]}, max_len:{Q.shape[1]}, num_heads:{self.num_heads}, lens of valid_lens:{valid_lens.shape[0]}")

                querys = self._transpose_qkv(self.W_Q(Q))
                keys = self._transpose_qkv(self.W_K(K))
                value = self._transpose_qkv(self.W_V(V))

                out = self._transpose_out(self.attention(querys, keys, value, valid_lens))
                return self.W_out(out)

        def _transpose_qkv(self, X: torch.Tensor) -> torch.Tensor:
                '''
                batch_size * max_len * num_hiddens --> batch_size * num_head * max_len * num_hiddens/num_head
                --> [ batch_size * num_head, max_len, num_hiddens/num_head ]
                '''
                shape = X.shape
                X = X.reshape(shape[0], shape[1], self.num_heads, self.num_hiddens_one_head)
                return X.permute(0, 2, 1, 3).reshape(-1, shape[1], self.num_hiddens_one_head)

        def _transpose_out(self, X: torch.Tensor) -> torch.Tensor:
                '''
                [ batch_size * num_head, max_len, num_hiddens/num_head ]
            --> [ batch_size,  num_head, max_len, num_hiddens/num_head ]
            --> [ batch_size, max_len, num_head, num_hiddens/num_head ]
            --> [ batch_size, max_len, num_hiddens ]
                '''
                shape = X.shape
                X = X.reshape(shape[0] // self.num_heads, self.num_heads, shape[1], self.num_hiddens_one_head)
                return X.permute(0, 2, 1, 3).reshape(-1, shape[1], self.num_heads * self.num_hiddens_one_head)

class PositionWiseFFN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, bias = False):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.ac = nn.ReLU()
        self.dense2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        return self.dense2(self.ac(self.dense1(X)))

class AddNorm(nn.Module):
    def __init__(self, in_features, dropout, bias = False):
        super(AddNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, X1 : torch.Tensor, X2 : torch.Tensor) -> torch.Tensor:
        return self.layer_norm(X1 + self.drop(X2))

class EncoderLayer(nn.Module):
    def __init__(self, query_size, key_size, val_size, num_heads, num_hiddens,
                       in_features, hidden_features, out_features,
                       dropout, index, bias = False):
        super(EncoderLayer, self).__init__()
        assert num_hiddens == in_features
        self.num_heads = num_heads
        self.index = index
        self.attention1 = MutilHeadAttention(query_size, key_size, val_size, num_heads, num_hiddens, dropout)
        self.add_norm1 = AddNorm(num_hiddens, dropout)
        self.ffn1 = PositionWiseFFN(in_features, hidden_features, out_features)
        self.add_norm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X : torch.Tensor, valid_lens : torch.Tensor) -> torch.Tensor:

        max_len = X.shape[1]
        if valid_lens is not None:
            enc_valid_lens = valid_lens.repeat_interleave(max_len).repeat_interleave(self.num_heads)

        Y1 = self.attention1(X, X, X, enc_valid_lens)
        Y2 = self.add_norm1(X, Y1)
        return self.add_norm2(Y2, self.ffn1(Y2))

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, query_size, key_size, val_size, num_heads, num_hiddens,
                       in_features, hidden_features, out_features, num_encoder_layers,
                       dropout, bias = False):
        super(TransformerEncoder, self).__init__()
        assert num_encoder_layers > 0
        self.embeding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_embeding_drop = nn.Dropout(dropout)
        self.layers = nn.Sequential()
        self.num_hiddens = num_hiddens
        for i in range(num_encoder_layers):
            self.layers.add_module(f"encoder_layer_{i}",
                      EncoderLayer(query_size, key_size, val_size, num_heads, num_hiddens,
                                   in_features, hidden_features, out_features, dropout, i))
    def forward(self, X : torch.Tensor, valid_lens : torch.Tensor) -> (torch.Tensor, torch.Tensor):
        assert X.dim() == 2
        X_emb = self.embeding(X)
        pos_embeding = postion_embeding(X_emb.shape[1], X_emb.shape[2]).to(X.device)
        # print(X_emb.shape)
        X_emb = self.pos_embeding_drop(X_emb / math.sqrt(X_emb.shape[2]) + pos_embeding)
        for layer in self.layers:
            X_emb = layer(X_emb, valid_lens)

        return X_emb, valid_lens


class DecoderLayer(nn.Module):
        def __init__(self, query_size, key_size, val_size, num_heads, num_hiddens,
                     in_features, hidden_features, out_features,
                     dropout, index, bias=False):
                super(DecoderLayer, self).__init__()
                assert num_hiddens == in_features
                self.index = index
                self.num_heads = num_heads
                self.self_attention = MutilHeadAttention(query_size, key_size, val_size, num_heads, num_hiddens,
                                                         dropout)
                self.add_norm1 = AddNorm(num_hiddens, dropout)

                self.encoder_decoder_attention = MutilHeadAttention(query_size, key_size, val_size, num_heads,
                                                                    num_hiddens, dropout)
                self.add_norm2 = AddNorm(num_hiddens, dropout)

                self.ffn = PositionWiseFFN(in_features, hidden_features, out_features)
                self.add_norm3 = AddNorm(num_hiddens, dropout)

        def forward(self, X: torch.Tensor, encoder_states: (torch.Tensor, torch.Tensor)) -> torch.Tensor:
                batch_size = X.shape[0]
                max_len = X.shape[1]
                encoder_output, encoder_valid_lens = encoder_states[0], encoder_states[1]
                # print(X.shape, encoder_output.shape, encoder_valid_lens.shape)
                if encoder_valid_lens is not None:
                        dec_valid_lens = torch.arange(1, 1 + max_len).repeat(batch_size).repeat(self.num_heads).to(
                                X.device)
                        # print(dec_valid_lens.shape[0])
                Y1 = self.add_norm1(X, self.self_attention(X, X, X, dec_valid_lens))

                enc_valid_lens = encoder_valid_lens.repeat_interleave(X.shape[1]).repeat_interleave(self.num_heads)
                Y2 = self.add_norm2(Y1,
                                    self.encoder_decoder_attention(Y1, encoder_output, encoder_output, enc_valid_lens))

                return self.add_norm3(Y2, self.ffn(Y2))


class TransformerDecoder(nn.Module):
        def __init__(self, vocab_size, query_size, key_size, val_size, num_heads, num_hiddens,
                     in_features, hidden_features, out_features, num_decoder_layers,
                     dropout, bias=False):
                super(TransformerDecoder, self).__init__()
                assert num_decoder_layers > 0
                self.embeding = nn.Embedding(vocab_size, num_hiddens)
                self.pos_embeding_drop = nn.Dropout(dropout)
                self.layers = nn.Sequential()
                self.num_hiddens = num_hiddens
                for i in range(num_decoder_layers):
                        self.layers.add_module(f"decoder_layer_{i}",
                                               DecoderLayer(query_size, key_size, val_size, num_heads, num_hiddens,
                                                            in_features, hidden_features, out_features, dropout, i))

                self.dense = nn.Linear(num_hiddens, vocab_size)

        def init_states(self, encoder_states):
                self.encoder_states = encoder_states

        def forward(self, X: torch.Tensor) -> torch.Tensor:
                assert X.dim() == 2
                X = self.embeding(X)
                pos_embeding = postion_embeding(X.shape[1], X.shape[2]).to(X.device)
                # print(X_emb.shape)
                X = self.pos_embeding_drop(X / math.sqrt(X.shape[2]) + pos_embeding)
                for layer in self.layers:
                        X = layer(X, self.encoder_states)

                return self.dense(X)

class Transformer(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, query_size, key_size, val_size, num_heads, num_hiddens,
                       in_features, hidden_features, out_features, num_encoder_layers, num_decoder_layers,
                       dropout, bias = False):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(enc_vocab_size, query_size, key_size, val_size, num_heads, num_hiddens,
                         in_features, hidden_features, out_features, num_encoder_layers, dropout)
        self.decoder = TransformerDecoder(dec_vocab_size, query_size, key_size, val_size, num_heads, num_hiddens,
                         in_features, hidden_features, out_features, num_decoder_layers, dropout)

    def forward(self, X, Y, enc_valid_lens):
        enc_output, enc_valid_lens = self.encoder(X, enc_valid_lens)
        self.decoder.init_states((enc_output, enc_valid_lens))
        return self.decoder(Y)

class MaskCrossEntropy(nn.CrossEntropyLoss):
    def __init__(self):
        super(MaskCrossEntropy, self).__init__(reduction="none")
    def forward(self, y_hat : torch.Tensor, y : torch.Tensor, valid_lens : torch.Tensor):
        loss_mat = super().forward(y_hat.transpose(2, 1), y)
        mask = sequence_mask(loss_mat.unsqueeze(2), valid_lens)
        # print(mask)
        loss = loss_mat.masked_fill(~mask, 0)
        # print(loss)
        return loss.mean()

if __name__ == '__main__':
        device = "cuda"
        vocab_size = 16
        query_size, key_size, val_size, num_heads, num_hiddens = 8, 8, 8, 2, 8
        in_features, hidden_features, out_features = 8, 16, 8
        dropout, num_decoder_layers, num_encoder_layers = 0.2, 2, 2
        model = Transformer(vocab_size, vocab_size, query_size, key_size, val_size, num_heads, num_hiddens,
                            in_features, hidden_features, out_features, num_encoder_layers, num_decoder_layers, dropout).to(device)
        X = torch.randint(0, 16, (2, 4)).to(device)
        enc_valid_lens = torch.Tensor([2, 3]).to(device)
        Y = torch.randint(0, 16, (2, 4)).to(device)

        y_hat = model(X, Y, enc_valid_lens)
        print(y_hat.shape)
        print(model.encoder.layers[0].attention1.attention.attention_weights)
        print(model.decoder.layers[0].self_attention.attention.attention_weights)
        print(model.decoder.layers[0].encoder_decoder_attention.attention.attention_weights)