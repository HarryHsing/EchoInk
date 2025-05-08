# Install the packages in r1-v .
cd src/r1-v 
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation

# vLLM support 
pip install vllm==0.7.2

pip install nltk
pip install rouge_score
pip install deepspeed

# latest version of transfoerms, otherwise Qwen2.5 Omni may not be included
pip uninstall transformers
pip install git+https://github.com/huggingface/transformers
pip install accelerate

# qwen video extraction setting, e.g., max frames, resolutions
# Use the [decord] feature to improve speed
cd src/qwen-vl-utils
pip install -e .[decord]
cd ../../

cd src/qwen-omni-utils
pip install -e .[decord]
cd ../../
