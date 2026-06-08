

# Audio Caption Generator

A deep learning pipeline that generates natural language captions from music and audio clips.

## Project Overview

This project converts audio recordings into descriptive text captions using a combination of:

* CLAP (Contrastive Language-Audio Pretraining) embeddings
* Classical audio features extracted with Librosa
* LSTM-based audio encoder
* T5 text generation decoder
* PyTorch deep learning framework

The model learns the relationship between audio content and textual descriptions, enabling automatic caption generation for unseen audio samples.

---

## Features

### Audio Augmentation

To improve dataset diversity, several augmentation techniques are applied:

* Pitch Shift (+2 semitones)
* Pitch Shift (-2 semitones)
* Time Stretch (Fast)
* Time Stretch (Slow)
* Gaussian Noise Injection

---

### Audio Feature Extraction

#### CLAP Embeddings

High-level semantic audio representations are extracted using LAION-CLAP.

#### Classical Audio Features

Additional handcrafted features are extracted using Librosa:

* Spectral Centroid
* Spectral Bandwidth
* Zero Crossing Rate
* RMS Energy
* Tempo

---

### Caption Generation Model

Architecture:

Audio Input
↓
CLAP Embedding + Classical Features
↓
Feature Normalization
↓
LSTM Encoder
↓
T5 Decoder
↓
Generated Caption

---

## Technologies Used

* Python
* PyTorch
* TorchVision
* Transformers (HuggingFace)
* LAION-CLAP
* Librosa
* Torchaudio
* NumPy
* Pandas
* Scikit-Learn
* NLTK
* ROUGE Score

---

## Training Strategy

* Transfer Learning with T5-Small
* CLAP Pretrained Audio Encoder
* Feature Fusion
* Early Stopping
* AdamW Optimizer
* BLEU Evaluation
* ROUGE Evaluation

---

## Evaluation Metrics

The model is evaluated using:

* BLEU Score
* ROUGE-1
* ROUGE-2
* ROUGE-L

---

## Dataset Pipeline

1. Audio collection
2. Audio augmentation
3. CLAP embedding extraction
4. Classical feature extraction
5. Feature normalization
6. Model training
7. Caption generation
8. Evaluation

---

## Example

Input:

Audio Clip (.wav)

Output:

"A calm instrumental piece with soft piano melodies and relaxing atmosphere."

---

## Future Improvements

* Transformer-based audio encoder
* Beam Search decoding
* CLAP fine-tuning
* Larger language models
* Multi-modal audio understanding
* Support for longer audio clips

---

## Author

Developed as a Deep Learning and Audio Understanding project using PyTorch, CLAP, and T5.
