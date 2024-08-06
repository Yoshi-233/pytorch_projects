import torch
import torch.nn as nn
from simple_code import TransformerModel
from d2l import torch as d2l

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps, device):
    """Predict for sequence to sequence.

    Defined in :numref:`sec_utils`"""
    # Set `net` to eval mode for inference
    net.eval()
    net.to(device)
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        # print(dec_X.shape)
        # print(len(net.decoder.encoder_states))
        Y = net.decoder(dec_X)
        # We use the token with the highest prediction likelihood as input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()

        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

if __name__ == '__main__':
        num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
        lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
        ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
        key_size, query_size, value_size = 32, 32, 32
        norm_shape = [32]

        _, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

        net = TransformerModel(len(src_vocab), len(tgt_vocab), tgt_vocab,
                               num_hiddens, ffn_num_hiddens, num_heads,
                               num_layers, num_layers, dropout)

        net.load_state_dict(torch.load('transformer.pt'))

        engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
        fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
        for eng, fra in zip(engs, fras):
                translation, dec_attention_weight_seq = predict_seq2seq(
                        net, eng, src_vocab, tgt_vocab, num_steps, device, False)
                print(f'{eng} => {translation}, label:{fra}',
                      f'bleu {d2l.bleu(translation, fra, k=2):.3f}')