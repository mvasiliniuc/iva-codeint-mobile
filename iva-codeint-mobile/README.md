# iva-codeint-mobile

## Overview
This is a personal effort to train and evaluate code generation models for mobile software development. IVA-codeint-swift and IVA-codeint-kotlin are GPT-2 models trained from scratch on Swift and Kotlin code respectively.
- initialize and train a GPT-2 (small) language model from scratch for code generation
- train a custom tokenizer adapted for Swift/Kotlin code
- cutate datasets with `datasets` library
- train with `accelerate` on multiple GPUs using data parallelism and mixed precision
- continuously push checkpoints to the hub with `huggingface_hub`
- stream the dataset with `datasets` during training to avoid disk bottlenecks
- uses `Weights & Biases` for experiment data gathering
- models can be found on HuggingFace: [Swift](https://huggingface.co/mvasiliniuc/iva-codeint-swift-small) [Kotlin](https://huggingface.co/mvasiliniuc/iva-codeint-kotlin-small) 

    
## Installation
To install the dependencies simply run the following command:
```bash
pip install -r requirements.txt
```

# Acknowledgement

Steps and approaches in this project used the [Codeparrot research project](https://github.com/huggingface/transformers/tree/main/examples/research_projects/codeparrot) as sample and technical guidance.
