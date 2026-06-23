---
library_name: peft
license: other
base_model: /root/autodl-tmp/models/Qwen2.5-7B-Instruct
tags:
- base_model:adapter:/root/autodl-tmp/models/Qwen2.5-7B-Instruct
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: disc_law_qwen7b_v3
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# disc_law_qwen7b_v3

This model is a fine-tuned version of [/root/autodl-tmp/models/Qwen2.5-7B-Instruct](https://huggingface.co//root/autodl-tmp/models/Qwen2.5-7B-Instruct) on the disc_law_labor_qa_v3, the disc_law_triplet_labor_qa_v3, the disc_law_eval_gold_v3 and the identity datasets.
It achieves the following results on the evaluation set:
- Loss: 0.9990

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 16
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.03
- num_epochs: 1.5

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.1105        | 0.3002 | 100  | 1.0580          |
| 1.1930        | 0.6004 | 200  | 1.0212          |
| 1.1091        | 0.9006 | 300  | 1.0043          |
| 1.0048        | 1.1981 | 400  | 0.9992          |
| 0.9892        | 1.4983 | 500  | 0.9991          |
| 0.9892        | 1.5013 | 501  | 0.9990          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.6.0
- Pytorch 2.12.0+cu130
- Datasets 4.0.0
- Tokenizers 0.22.2