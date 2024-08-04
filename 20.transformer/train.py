import torch
from model import Transformer, MaskCrossEntropy
import torch.nn as nn
from d2l import torch as d2l

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
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
    loss = MaskCrossEntropy()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')

if __name__ == '__main__':
        num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
        lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
        ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
        key_size, query_size, value_size = 32, 32, 32
        norm_shape = [32]

        train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
        net = Transformer(len(src_vocab), len(tgt_vocab),
                          key_size, query_size, value_size, num_heads, num_hiddens,
                          ffn_num_input, ffn_num_hiddens, num_hiddens,
                          num_layers, num_layers, dropout)
        train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
        torch.save(net.state_dict(), './model.pt')
