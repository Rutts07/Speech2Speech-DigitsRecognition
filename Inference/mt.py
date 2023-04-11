# Contains the definition of the encoder, decoder and seq2seq model
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from vocab_mt import SOS_token, EOS_token, UNK_token

# Define the encoder
# Encoder inputs : One-hot encoded ASR labels (B X L X U)
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size                    # U
        self.hidden_size = hidden_size                  # H
        self.n_layers = n_layers                        
        
        self.U = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
        
    def _init_hidden(self, batch_size):
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)
    
    # Train the encoder with the decoder
    def forward(self, input, hidden=None):
        hidden = self._init_hidden(input.size(0)) if hidden is None else hidden
                                                        # 2 x B x H
        input = self.U(input)                           # B x L x H
        outputs, hidden = self.gru(input, hidden)       # B x L x H, 2 x B x H
        
        hidden = hidden[-1].unsqueeze(0)                # 1 x B x H
        outputs = outputs[:, -1, :].unsqueeze(1)        # B x 1 x H
        
        return outputs, hidden                          # B x L x H, 1 x B x H                        
    
    # Encoder in inference mode
    def inference(self, input):
        hidden = self._init_hidden(input.size(0))
        input = self.U(input)
        outputs, hidden = self.gru(input, hidden)
        
        hidden = hidden[-1].unsqueeze(0)
        outputs = outputs[:, -1, :].unsqueeze(1)
        
        return outputs, hidden

# Define the encoder-decoder attention model
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V  = nn.Linear(hidden_size, 1)
        
    def attend(self, hidden, encoder_states):
        # hidden : 1 x B x H
        # encoder_outputs : B x L x H
        
        hidden = hidden.squeeze(0)                      # B x H
        hidden = hidden.unsqueeze(1)                    # B x 1 x H
        
        # Calculate the attention weights
        attn_weights = self.V(torch.tanh(self.W1(hidden) + self.W2(encoder_states)))
        attn_weights = F.softmax(attn_weights, dim=1)   # B x L x 1
        
        return attn_weights
    
# Define the decoder
# Decoder inputs : Encoder outputs (B x L x H)
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size                  # H
        self.output_size = output_size                  # V
        
        self.U = nn.Linear(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, 1, batch_first=True)
        self.V = nn.Linear(hidden_size, output_size)
        self.attn = Attention(hidden_size)
    
    # Decoder in training mode  
    def forward(self, input, last_hidden, encoder_states):  
        input = self.U(input)                                       # B x 1 x H        
        atn_weights = self.attn.attend(last_hidden, encoder_states) # B x L x 1
        context = atn_weights * encoder_states                      # B x L x 1 * B x L x H = B x L x H
        
        gru_input = torch.cat((input, context), dim=1)              # B x (L+1) x H
        gru_input = torch.sum(gru_input, dim=1).unsqueeze(1)        # B x 1 x H
        
        output, hidden = self.gru(gru_input, last_hidden)           # B x 1 x H, B x 1 x H        
        output = self.V(output)                                     # B x 1 x V
        
        return output, hidden
    
    # Decoder in inference mode
    def inference(self, input, last_hidden, encoder_states):
        input = self.U(input)                                       # 1 x 1 x H         
        atn_weights = self.attn.attend(last_hidden, encoder_states) # 1 x L x 1
        context = atn_weights * encoder_states                      # 1 x L x 1 * 1 x L x H = 1 x L x H
        
        gru_input = torch.cat((input, context), dim=1)              # B x (L+1) x H
        gru_input = torch.sum(gru_input, dim=1).unsqueeze(1)        # B x 1 x H
        
        output, hidden = self.gru(gru_input, last_hidden)           # 1 x 1 x H, 1 x 1 x H
        output = self.V(output)                                     # 1 x 1 x V
        labels = F.softmax(output, dim=2)                           # 1 x 1 x V
        
        return output, labels, hidden
    
# Define the se2seq model
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=2):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderRNN(input_size, hidden_size, n_layers)
        self.decoder = DecoderRNN(hidden_size, output_size)
        
    def forward(self, inputs, targets, teacher_forcing_ratio=0.5):
        batch_size = inputs.size(0)
        vocab_size = self.decoder.output_size
        
        encoder_states, hidden = self.encoder(inputs)
        
        target_len = targets.size(1)
        outputs = torch.zeros(batch_size, target_len, vocab_size)
        decoder_input = torch.zeros(batch_size, 1, vocab_size)
        decoder_input[:, 0, SOS_token] = 1
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        if use_teacher_forcing:
            for t in range(target_len):
                output, hidden = self.decoder(decoder_input, hidden, encoder_states)
                outputs[:, t, :] = output.squeeze(1)
                decoder_input = targets[:, t, :].unsqueeze(1)
        else:
            for t in range(target_len):
                output, hidden = self.decoder(decoder_input, hidden, encoder_states)
                outputs[:, t, :] = output.squeeze(1)
                topv, topi = output.topk(1)
                
                decoder_input = torch.zeros(batch_size, 1, vocab_size)
                decoder_input[:, 0, topi] = 1
        
        return outputs
    
    def predict(self, input):
        batch_size = input.size(0)
        target_len = input.size(1)
        vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, target_len, vocab_size)
        encoder_states, hidden = self.encoder(input)
        
        decoder_input = torch.zeros(batch_size, 1, vocab_size)
        decoder_input[:, 0, SOS_token] = 1
        
        for t in range(target_len - 1):
            output, labels, hidden = self.decoder.inference(decoder_input, hidden, encoder_states)
            outputs[:, t, :] = labels.squeeze(1)
            topv, topi = labels.topk(1)
            decoder_input = torch.zeros(batch_size, 1, vocab_size)
            decoder_input[:, 0, topi] = 1
        
        # Add the EOS token
        outputs[:, target_len - 1, EOS_token] = 1
        
        return outputs
    
# Inference
def mt_inference(seq2seq, input, vocab_asr, vocab_mt, max_len=7):    
    # truncate the input to max_len of training samples and remove the unknown tokens
    input = input[:max_len]  
    input = [token for token in input if vocab_asr.dig2idx(token) != UNK_token]                        
    encoded_seq = [SOS_token] + vocab_asr.encode(input) + [EOS_token]
    
    # one-hot encode the input
    one_hot_enc = np.zeros((1, len(encoded_seq), vocab_asr.vocab_size()))
    for i, token in enumerate(encoded_seq):
        one_hot_enc[0, i, token] = 1
        
    one_hot_enc = torch.tensor(one_hot_enc, dtype=torch.float)
    decoded_seq = seq2seq.predict(one_hot_enc)
    decoded_seq = torch.argmax(decoded_seq, dim=2).flatten().tolist()
    
    decoded_seq = [token for token in decoded_seq if token not in [SOS_token, EOS_token]]
    
    return vocab_mt.decode(decoded_seq)

# Inference
"""
from vocab_mt import ASR_Vocab, MT_Vocab
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Rule based src to tgt mapping
def translate(text):
    out = ""
    for i in range(len(text)):
        if text[i] == '1': out = out + "ஒன்று "
        elif text[i] == '2': out = out + "இரண்டு "
        elif text[i] == '3': out = out + "மூன்று "
        elif text[i] == '4': out = out + "நான்கு "
        elif text[i] == '5': out = out + "ஐந்து "
        elif text[i] == '6': out = out + "ஆறு "
        elif text[i] == '7': out = out + "ஏழு "
        elif text[i] == '8': out = out + "எட்டு "
        elif text[i] == '9': out = out + "ஒன்பது "
        else: out = out + "பூஜ்யம் "
        
    out = out[:-1]
    return out

# Calculate the BLEU score
def bleu_scores(reference, hypothesis):
    score = corpus_bleu([[reference]], [hypothesis], smoothing_function=SmoothingFunction().method1)
    return score

# Create the MT_ASR vocabulary
with open('./data/ASR/ASR_Labels.txt', 'r') as f:
    asr_labels = f.read().splitlines()
asr_labels = [asr_label[:-1] for asr_label in asr_labels]

vocab_asr = ASR_Vocab(asr_labels)
vocab_asr.build_vocab()

# Create the MT vocabulary
with open('./data/ASR/MT_Labels.txt', 'r') as f:
    mt_labels = f.read().splitlines()
mt_labels = [mt_label[:-1] for mt_label in mt_labels]

vocab_mt = MT_Vocab(mt_labels)
vocab_mt.build_vocab()

input_seq = 'O12345Z'
if len(sys.argv) > 1:
    input_seq = ""
    for i in sys.argv[1]:
        if vocab_asr.dig2idx(i) != UNK_token:
            input_seq += i
        
# Load the model
mt_model = Seq2Seq(vocab_asr.vocab_size(), 300, vocab_mt.vocab_size())
mt_model.load_state_dict(torch.load('./models/seq2seq.pth'))

# Test the model
mt_model.eval()
output_seq = mt_inference(mt_model, input_seq, vocab_asr, vocab_mt)
print('Input      : ', input_seq)
print('Reference  : ', translate(input_seq))
print('Hypothesis : ', output_seq)
print('BLEU score : ', bleu_scores(translate(input_seq), output_seq))
"""