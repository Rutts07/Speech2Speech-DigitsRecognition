# Code to calculate CER and BLEU scores for the SSMT system
import os
import torch, torchaudio
import pandas as pd
from vocab_asr import ASR_Vocab
from asr import CTC_ASR, asr_inference, ctc_decode
from torchmetrics.functional import char_error_rate
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

print('Creating vocabularies and loading models...')

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

##### Load MT model and generate MT results
from vocab_mt import ASR_Vocab, MT_Vocab
from mt import Seq2Seq, mt_inference
    
# Create the MT_ASR vocabulary
with open('./data/ASR/ASR_Labels.txt', 'r') as f:
    asr_labels = f.read().splitlines()
asr_labels = [asr_label[:-1] for asr_label in asr_labels]

vocab_asr_mt = ASR_Vocab(asr_labels)
vocab_asr_mt.build_vocab()

# Create the MT vocabulary
with open('./data/ASR/MT_Labels.txt', 'r') as f:
    mt_labels = f.read().splitlines()
mt_labels = [mt_label[:-1] for mt_label in mt_labels]

vocab_mt = MT_Vocab(mt_labels)
vocab_mt.build_vocab()

# Define the Seq2Seq model parameters
input_size  = vocab_asr_mt.vocab_size()
hidden_size = 300
output_size = vocab_mt.vocab_size()

# Load the seq2seq model
mt_model = Seq2Seq(input_size, hidden_size, output_size)
mt_model.load_state_dict(torch.load('./models/seq2seq.pth'))

# Preprocess the audio files
def preprocess_audio(audio_file, path='./data/ASR/test/'):
    waveform, sample_rate = torchaudio.load(path + audio_file)

    # Resample the audio file to 16kHz and convert to mono channel
    if sample_rate != 16000: waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    waveform = waveform.mean(dim=0, keepdim=True).squeeze(0)
    
    return waveform

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

# Calculate the CER and BLEU scores
def wer_score(reference, hypothesis):
    score = char_error_rate(hypothesis, reference)
    score = score.detach().cpu().numpy()
    return score
            
# Calculate the BLEU score
def bleu_scores(reference, hypothesis):
    score = corpus_bleu([[reference]], [hypothesis], smoothing_function=SmoothingFunction().method1)
    return score

# Set the models to evaluation mode
asr_model.eval()
mt_model.eval()

# Load the audio files from test set
audio_files = os.listdir('./data/ASR/test')
audio_files = [audio_file for audio_file in audio_files if audio_file.endswith('.wav')]

num_files = len(audio_files)
audio_files = audio_files[:num_files]

wer, blu = 0.0, 0.0
mts_references, mts_hypotheses = [], []

for i in range(num_files):
    src_transcript = audio_files[i].split('_')[0][:-1]
    asr_transcript = ctc_decode(asr_inference(asr_model, mel_spectrogram, preprocess_audio(audio_files[i])), vocab_asr)
    wer += wer_score(src_transcript, asr_transcript)
    """
    print("ACT Transcript : ", src_transcript)
    print("ASR Transcript : ", asr_transcript)
    print("WER : ", wer_score(src_transcript, asr_transcript))
    """
    
    src_translated = translate(src_transcript)
    mts_translated = mt_inference(mt_model, asr_transcript, vocab_asr_mt, vocab_mt)
    mts_references.append([src_translated.split()])
    mts_hypotheses.append(mts_translated.split())
    blu = bleu_scores(src_translated, mts_translated)
    """
    print("ACT Translation : ", src_translated)
    print("MTS Translation : ", mts_translated)
    print("BLEU : ", blu)
    print() 
    """   
    
    # Print the progress
    if (i + 1) == num_files: print('\033[0K\r{}/{} files processed ...'.format(i + 1, num_files), end='')
    elif (i + 1) % 50 == 0: print('\033[0K\r{}/{} files processed ...'.format(i + 1, num_files), end='')
    
wer = wer / num_files * 1.0
blu = corpus_bleu(mts_references, mts_hypotheses, smoothing_function=SmoothingFunction().method1, weights=(0.5, 0.5))

print("\nAverage WER : {:.2f}".format(wer))
print("Average BLEU: {:.2f}".format(blu))
    