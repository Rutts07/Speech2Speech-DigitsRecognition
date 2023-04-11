# Speech2Speech-DigitsRecognition
This repository contains the code for a speech to speech translation system created from scratch for digits translation from English to Tamil.

#### Framework :
- All the models were built using ```PyTorch```.
- All the models were trained on a ```NVIDIA GeForce RTX 2070 Super with Max-Q Design``` GPU.

#### Pipeline : 
- Automatic Speech Recognition (ASR) :
    - A Connectionist Temporal Classification (CTC) based ASR.
    - Based on a Recurrent Neural Network (RNN) on top of a Convolutional Neural Network (CNN) which is used obtain features from the raw audio signal.
    - The RNN is a Bidirectional Gate Recurrent Unit (GRU) while the CNN is a 1D Convolutional Neural Network with max pooling and batch normalization.
    - The CTC loss was used to train the model.
    - The model was trained for 100 epochs with a batch size of 64.

- Machine Translation (MT) :
    - A Vanilla Sequence to Sequence (Seq2Seq) model with Attention.
    - The Encoder is a Bidirectional GRU while the Decoder is a GRU.
    - The Attention mechanism is a Bahdanau Attention using simple MLPs.
    - Cross Entropy loss was used to train the model.
    - The model was trained for 300 epochs with a batch size of 512.

- Text to Speech (TTS) :
    - Google API was used to convert the text to speech.

#### Performance :
- ASR (Model : CTC-GRU-CNN) (Metric : CER) :
    - The model was able to achieve a CER of 0.04 on the test set.

- MT (Model : Seq2Seq-Attention) (Metric : BLEU) :
    - The model was able to achieve a BLEU score of 0.82 on the test set.
    