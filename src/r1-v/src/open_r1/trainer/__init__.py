# from .grpo_trainer import Qwen2VLGRPOTrainer, Qwen2_5OmniGRPOTrainer
from .grpo_trainer import Qwen2_5OmniGRPOTrainer
from .vllm_grpo_trainer_modified import Qwen2VLGRPOVLLMTrainerModified

__all__ = [
    # "Qwen2VLGRPOTrainer", 
    # "Qwen2VLGRPOVLLMTrainerModified",
    "Qwen2_5OmniGRPOTrainer",
]
