cd src/r1-v

export DEBUG_MODE="true" 
export LOG_PATH="./debug_log_avqa-r1.txt"
export WANDB_NAME=TEST
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12366" \
    src/open_r1/grpo.py \
    --output_dir ./log/$WANDB_NAME \
    --model_name_or_path ../../../models/Qwen2.5-Omni-7B \
    --dataset_name ../../../datasets/AVQA_R1/train/omni_rl_format_train.json \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --len_control false \
    --weighted_reward false\
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --save_steps 100 \
    --beta 0.001 \
    --max_grad_norm 5 \
    --save_only_model true \
    --num_generations 16 \
    --model_type omni \
    --use_audio_in_video true \