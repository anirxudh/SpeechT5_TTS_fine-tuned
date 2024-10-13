---
library_name: transformers
language:
- nl
license: mit
base_model: microsoft/speecht5_tts
tags:
- generated_from_trainer
datasets:
- Yassmen/TTS_English_Technical_data
model-index:
- name: SpeechT5 fine-tuning
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# SpeechT5 fine-tuning

This model is a fine-tuned version of [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) on the PeekieTech data dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4572

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

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
