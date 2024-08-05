import torch
import torch.nn as nn
from d2l import torch as d2l
import random
import numpy as np
import math


class TransformerModel(nn.Module):
        def __init__(self, src_vocab_size, tgt_vocab_size, tgt_vocab, d_model, ffn_num_hiddens, nhead,
                     num_encoder_layers, num_decoder_layers, dropout):
                super().__init__()
                self.d_model = d_model
                self.nhead = nhead
                self.tgt_vocab = tgt_vocab
                print(src_vocab_size, tgt_vocab_size)
                self.src_embedding = nn.Embedding(src_vocab_size, d_model)
                self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
                self.pos_embedding_dropout = nn.Dropout(dropout)
                self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, dim_feedforward=ffn_num_hiddens,
                                                  num_encoder_layers=num_encoder_layers,
                                                  num_decoder_layers=num_decoder_layers,
                                                  batch_first=True, dropout=dropout)
                self.out = nn.Linear(d_model, tgt_vocab_size)
                self.loss = nn.CrossEntropyLoss(reduction='none')

        def forward(self, src, src_valid_lens, tgt, tgt_valid_lens):
                assert src.size() == tgt.size()
                assert src_valid_lens.size() == tgt_valid_lens.size()

                batch_size = src.size(0)
                max_len = src.size(1)

                src = self.src_embedding(src)
                src_pos = self.position_embeding(max_len, self.d_model).to(src.device)
                src = self.pos_embedding_dropout(src / math.sqrt(self.d_model) + src_pos)

                bos = torch.tensor([self.tgt_vocab['<bos>']] * batch_size,
                                   device=tgt.device).reshape(-1, 1)
                dec_input = d2l.concat([bos, tgt[:, :-1]], 1)  # Teacher forcing
                dec_input = self.tgt_embedding(dec_input)
                dec_input_pos = self.position_embeding(max_len, self.d_model).to(dec_input.device)
                dec_input = self.pos_embedding_dropout(dec_input / math.sqrt(self.d_model) + dec_input_pos)

                src_mask = TransformerModel.attention_mask(src_valid_lens, max_len, self.nhead, type="bool")
                tgt_mask = TransformerModel.square_subsequent_mask(batch_size, max_len, self.nhead).to(tgt.device)
                # print(src.device, dec_input.device, src_mask.device, tgt_mask.device)
                output = self.transformer(src, dec_input, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=src_mask)
                output = self.out(output)

                # print(output.shape, tgt.shape)
                loss_mat = self.loss(output.transpose(2, 1), tgt)
                loss_mask = self.key_mask(tgt_valid_lens, max_len)
                # print(loss_mask)
                loss = loss_mat.masked_fill(~loss_mask, 0)
                return loss.mean()
        def predict(self, src, src_valid_lens, max_len):
                pass

        @staticmethod
        def attention_mask(valid_lens, max_len, num_heads, type="bool"):
                assert type in ["bool", "float"]
                masks = torch.arange(max_len, device=valid_lens.device)[None, :] < valid_lens.repeat_interleave(
                        max_len)[:, None]
                # masks = masks.to(valid_lens.device)

                if type == "bool":
                        return (~masks).repeat_interleave(num_heads, dim=0).reshape(-1, max_len, max_len)
                else:
                        masks = torch.zeros_like(masks).masked_fill(~masks, float('-inf'))
                        return (~masks).repeat_interleave(num_heads, dim=0).reshape(-1, max_len, max_len)

        @staticmethod
        def key_mask(valid_lens, max_len):
                masks = torch.ones(valid_lens.shape[0], max_len, dtype=torch.bool)

                for i, len in enumerate(valid_lens):
                        masks[i, :len] = False

                return masks.to(valid_lens.device)

        @staticmethod
        def square_subsequent_mask(batch_size, max_len, num_heads):
                return nn.Transformer.generate_square_subsequent_mask(max_len).to(dtype=torch.bool).repeat(
                        batch_size * num_heads, 1, 1)

        @staticmethod
        def position_embeding(max_len, dim) -> torch.Tensor:
                position_embeding = torch.zeros((max_len, dim))

                position_mat = torch.arange(max_len)[:, None] / torch.pow(10000,
                                                                          torch.arange(0, dim, 2) / dim)[None, :]

                position_embeding[:, 0::2] = torch.sin(position_mat)
                position_embeding[:, 1::2] = torch.cos(position_mat)

                return position_embeding


def train_seq2seq(net, data_iter, lr, num_epochs, device):
        """Train a model for sequence to sequence.

        Defined in :numref:`sec_utils`"""

        def xavier_init_weights(m):
                if type(m) == nn.Linear:
                        nn.init.xavier_uniform_(m.weight)
                if type(m) == nn.GRU:
                        for param in m._flat_weights_names:
                                if "weight" in param:
                                        nn.init.xavier_uniform_(m._parameters[param])

        net.apply(xavier_init_weights)
        net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        net.train()
        # animator = d2l.Animator(xlabel='epoch', ylabel='loss',
        #                         xlim=[10, num_epochs])
        for epoch in range(num_epochs):
                timer = d2l.Timer()
                metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
                for batch in data_iter:
                        optimizer.zero_grad()
                        X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
                        l = net(X, X_valid_len, Y, Y_valid_len)
                        print(l)
                        l.sum().backward()  # Make the loss scalar for `backward`

                        d2l.grad_clipping(net, 1)
                        num_tokens = Y_valid_len.sum()
                        optimizer.step()
                        with torch.no_grad():
                                metric.add(l.sum(), num_tokens)
                if (epoch + 1) % 10 == 0:
                        # animator.add(epoch + 1, (metric[0] / metric[1],))
                        print(f'epoch {epoch + 1}, loss {l}, ')
        print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
              f'tokens/sec on {str(device)}')


if __name__ == '__main__':
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Example usage
        # src_mask = TransformerModel.attention_mask(torch.tensor([2, 3]), 5, 2, type="bool")
        # key_padding_mask = TransformerModel.key_mask(torch.tensor([2, 3]), 5)
        # target_mask = TransformerModel.square_subsequent_mask(2, 5, 2)
        # print(target_mask)
        #
        # X = torch.randn(2, 5, 6)
        # tgt = torch.rand(2, 5, 6)
        # model = nn.Transformer(d_model=6, nhead=2, num_encoder_layers=2, num_decoder_layers=2, batch_first=True)
        # output = model(src=X, tgt=tgt, src_mask = src_mask, tgt_mask=target_mask, memory_mask=src_mask)
        # print(output)

        num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
        lr, num_epochs, device = 0.004, 200, d2l.try_gpu()
        ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
        key_size, query_size, value_size = 32, 32, 32

        train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
        net = TransformerModel(len(src_vocab), len(tgt_vocab), tgt_vocab,
                               num_hiddens, ffn_num_hiddens, num_heads,
                               num_layers, num_layers, dropout)
        # print(net(*next(iter(train_iter))))
        train_seq2seq(net, train_iter, lr, 2, device)
        torch.save(net.state_dict(), './transformer.pt')
