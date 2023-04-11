##### Load ASR model and generate ASR results
import sys, os
import torch, torchaudio
import pandas as pd
from vocab_asr import ASR_Vocab
from asr import CTC_ASR, asr_inference, ctc_decode

# Create a dictionary of audio files and their corresponding ASR labels
audio_df = pd.read_csv('./data/ASR/data.txt', sep=",")   
asr_labels = audio_df['Transcription'].values

vocab_asr = ASR_Vocab(asr_labels)
vocab_asr.build_vocab()

# Initialize the MFCC filter
n_mels = 32
mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=16000, 
    n_fft=512, win_length=512, hop_length=256, n_mels=n_mels, 
    f_min=0.0, f_max=None, pad=0, power=2.0, normalized=False)

# Define the CTC_ASR model parameters
input_size  = n_mels
hidden_size = 300
output_size = vocab_asr.vocab_size()

# Load the CTC_ASR model
asr_model = CTC_ASR(input_size, hidden_size, output_size)
asr_model.load_state_dict(torch.load('./models/ctc_asr.pth'))

# Load the audio file from command line argument
asr_model.eval()
if len(sys.argv) > 1: audio_file = sys.argv[1]
else : audio_file = './test.wav'

waveform, sample_rate = torchaudio.load(audio_file)

# Resample the audio file to 16kHz and convert to mono channel
if sample_rate != 16000: waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
waveform = waveform.mean(dim=0, keepdim=True).squeeze(0)

asr_output = ctc_decode(asr_inference(asr_model, mel_spectrogram, waveform), vocab_asr)
print('ASR output: ', asr_output)

##### Load MT model and generate MT results
from vocab_mt import ASR_Vocab, MT_Vocab
from mt import Seq2Seq, mt_inference
    
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

# Define the Seq2Seq model parameters
input_size  = vocab_asr.vocab_size()
hidden_size = 300
output_size = vocab_mt.vocab_size()

# Load the seq2seq model
mt_model = Seq2Seq(input_size, hidden_size, output_size)
mt_model.load_state_dict(torch.load('./models/seq2seq.pth'))

# Test the model
mt_model.eval()
output_seq = mt_inference(mt_model, asr_output, vocab_asr, vocab_mt)
print('MT output: ', output_seq)

##### Load the TTS API and generate the audio file
import os, gtts as gt
from pydub import AudioSegment
from pydub.playback import play
   
tts = gt.gTTS(text=output_seq, lang='ta')
tts.save("test.mp3")

choice = input("Do you want to play the audio? (y/n) : ")

if choice == 'y' or choice == 'Y':
    sound = AudioSegment.from_mp3("test.mp3")
    play(sound)
    os.remove("test.mp3")
