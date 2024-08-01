import transformer
import torch
from d2l import torch as d2l

if __name__ == '__main__':
        num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
        lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
        ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
        key_size, query_size, value_size = 32, 32, 32
        norm_shape = [32]

        train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

        encoder = transformer.TransformerEncoder(
                len(src_vocab), key_size, query_size, value_size, num_hiddens,
                norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                num_layers, dropout)
        decoder = transformer.TransformerDecoder(
                len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
                norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                num_layers, dropout)
        net = transformer.EncoderDecoder(encoder, decoder)
        transformer.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
        torch.save(net.state_dict(), './model.pt')