# Contains the definition of the CTC based ASR model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Define the CTC based ASR model
class CTC_ASR(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CTC_ASR, self).__init__()
        self.input_size = input_size        # U
        self.hidden_size = hidden_size      # H
        self.output_size = output_size      # V
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_size, input_size, 10, 2, 5)
        self.maxp1 = nn.MaxPool1d(2, 2)
        self.batn1 = nn.BatchNorm1d(input_size)
        
        # GRU layer
        self.U = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.W = nn.Linear(hidden_size, output_size)
        
    def _init_hidden(self, batch_size):
        return torch.zeros(4, batch_size, self.hidden_size)
        
    def forward(self, inputs):
        hidden = self._init_hidden(inputs.shape[0])
        
        inputs = inputs.transpose(1, 2)                                 # B x U x L
        cnn_out = F.gelu(self.batn1(self.conv1(inputs)))                # B x U x L'        
        cnn_out = cnn_out.transpose(1, 2)                               # B x L' x U
        
        gru_in = F.relu(self.U(cnn_out))                                # B x L' x H
        outputs, hidden = self.gru(gru_in, hidden)                      # B x L' x 2H, 4 x B x H
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:] 
                                                                        # B x L' x H
        
        outputs = self.W(outputs)                                       # B x L' x V
        # print(outputs.shape, hidden.shape)
        
        return outputs
    
    def predict(self, inputs):
        outputs = self.forward(inputs)
        outputs = F.softmax(outputs, dim=2)
        
        return torch.argmax(outputs, dim=2)
    
# Inference
def asr_inference(model, mel_spectrogram, waveform, max_feat_len=322):
    # Obtain and normalize the MFCCs
    mel_spec = mel_spectrogram(waveform)
    mel_spec = abs(mel_spec)
    mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
    
    mel_spec.unsqueeze_(0)
    if mel_spec.shape[2] > max_feat_len:
        mel_spec = mel_spec[:, :, :max_feat_len]
    
    # Pad the MFCCs to the maximum length
    else:
        mel_spec = F.pad(mel_spec, (0, max_feat_len - mel_spec.shape[2]), "constant", 0)

    mel_spec = mel_spec.transpose(1, 2)
    
    # Predict the output
    model.eval()
    output = model.predict(mel_spec).squeeze(0)
    output = output.tolist()
    
    return output

# Decode the output
def ctc_decode(output, vocab_asr):
    decoded_output = []
    prev = -1
    for i in range(len(output)):
        if output[i] != prev:
            decoded_output.append(output[i]) 
        prev = output[i]
    
    decoded_output = [x for x in decoded_output if x != 0 and x != 1]
    return vocab_asr.decode(decoded_output)
