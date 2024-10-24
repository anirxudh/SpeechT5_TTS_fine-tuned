# SpeechT5 fine-tuning


![Banner](https://github.com/anirxudh/Python/blob/main/speech-to-text.jpg)
---
library_name: transformers
language:
- en
license: mit
base_model: microsoft/speecht5_tts
tags:
- generated_from_trainer
datasets:
- Yassmen/TTS_English_Technical_data
model-index:
- name: SpeechT5 fine-tuning
  results: [accurate result]
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->



This model is a fine-tuned version of [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) on the PeekieTech data dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4572

# SpeechT5 Model Fine-tuned for Technical Vocabulary

## Model Description
The SpeechT5 model is a versatile speech processing model designed for tasks such as text-to-speech (TTS), automatic speech recognition (ASR), and other speech-related applications. This version of the model has been **fine-tuned specifically for technical vocabulary in English**, making it well-suited for domains where specialized terms are frequently used (e.g., engineering, computer science, medical terminology).

The fine-tuned model has been trained to:
- Accurately recognize and synthesize **technical vocabulary** in English.
- Improve the pronunciation and understanding of complex, domain-specific terms, ensuring higher accuracy in tasks like speech synthesis and recognition for technical contexts.

## Intended Uses & Limitations

### Intended Uses:
- **Text-to-Speech (TTS)**: The model can convert technical English text (especially in domains such as computer science, engineering, and medical fields) into accurate and natural-sounding speech.
- **Automatic Speech Recognition (ASR)**: This model can transcribe technical speech, even when containing complex vocabulary, with higher accuracy than general-purpose ASR systems.

### Limitations:
- **Non-technical Vocabulary**: While the model excels at technical speech and vocabulary, its performance on more casual or conversational English might be less optimized.
- **Generalization**: The model may not perform as well on very niche or rare technical terms not present in the training data.
- **Language Restriction**: This fine-tuned version is specific to **English technical vocabulary** and is not trained to generalize across other languages.

## Training and Evaluation Data
- **Training Dataset**: The model was fine-tuned using the `Yassmen/TTS_English_Technical_data` dataset. This dataset is specifically designed to cover a broad range of technical vocabulary in English, with audio-text pairs that include complex terms from various technical domains like IT, engineering, and medicine.
  - **Size**: It contains 1.94 GB of speech data.
  - **Content**: The dataset focuses on highly specialized vocabulary and jargon across multiple fields.

## Training Procedure
1. **Preprocessing**: The input data was preprocessed to ensure that both the text and audio were properly aligned, with steps such as text normalization (especially for technical terms) and feature extraction from audio samples.
   - **Text Normalization**: Special care was taken to normalize abbreviations, symbols, and numerical expressions common in technical speech.
   - **Audio Preprocessing**: Spectrograms were generated from raw audio files using a standard feature extraction pipeline.

2. **Model Architecture**: SpeechT5's architecture was leveraged for its ability to handle both ASR and TTS tasks effectively. The fine-tuning process focused on adapting the output layer to better handle technical terms.

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 32
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 4000
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch   | Step | Validation Loss |
|:-------------:|:-------:|:----:|:---------------:|
| 0.5281        | 3.5778  | 1000 | 0.4781          |
| 0.4908        | 7.1556  | 2000 | 0.4661          |
| 0.4972        | 10.7335 | 3000 | 0.4595          |
| 0.4858        | 14.3113 | 4000 | 0.4572          |


### Framework versions

- Transformers 4.46.0.dev0
- Pytorch 2.4.1+cu121
- Datasets 3.0.1
- Tokenizers 0.20.1
