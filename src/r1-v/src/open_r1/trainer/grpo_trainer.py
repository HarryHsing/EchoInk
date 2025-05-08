# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import random

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AutoConfig,  # 添加这一行
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from qwen_vl_utils import process_vision_info
from qwen_omni_utils import process_mm_info, process_audio_info, process_vision_info

import copy
from IPython import embed
import re

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
    

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class Qwen2_5OmniGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method with Qwen2.5Omni model support.
    This extends the GRPO algorithm to handle multimodal inputs including audio and video.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        use_audio_in_video: bool = True,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
            
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            
            # 在 Qwen2_5OmniGRPOTrainer.__init__ 方法中的模型初始化部分
            if "Qwen2.5-Omni" in model_id:
                # 为 Qwen2.5-Omni 模型创建一个新的 model_init_kwargs 字典
                omni_model_init_kwargs = model_init_kwargs.copy()
                if "use_cache" in omni_model_init_kwargs:
                    del omni_model_init_kwargs["use_cache"]
                
                # 加载配置并修改
                config = AutoConfig.from_pretrained(model_id)
                config.enable_audio_output = False  # 在配置中禁用音频输出
                
                # 使用修改后的配置加载模型
                model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    model_id, 
                    config=config,
                    **omni_model_init_kwargs
                )

                # 确保 talker 被禁用
                model.disable_talker()
                print("Talker component has been disabled")
            else:
                raise ValueError(f"Unsupported model ID for Qwen2.5OmniGRPOTrainer: {model_id}")
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # 在创建参考模型部分
        if is_deepspeed_zero3_enabled():
            if "Qwen2.5-Omni" in model_id:
                # 同样处理参考模型
                omni_model_init_kwargs = model_init_kwargs.copy()
                if "use_cache" in omni_model_init_kwargs:
                    del omni_model_init_kwargs["use_cache"]
                
                config = AutoConfig.from_pretrained(model_id)
                config.enable_audio_output = False  # 在配置中禁用音频输出
                
                self.ref_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    model_id, 
                    config=config,
                    **omni_model_init_kwargs
                )
                self.ref_model.disable_talker()
                print("Reference model talker component has been disabled")
            else:
                raise ValueError(f"Unsupported model ID for reference model: {model_id}")
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        if processing_class is None:
            if "Qwen2.5-Omni" in model_id:
                processing_class = Qwen2_5OmniProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                raise ValueError(f"Unsupported model ID for processor: {model_id}")

        # Save audio processing flag
        self.use_audio_in_video = use_audio_in_video

        # The rest of the initialization is the same as in Qwen2VLGRPOTrainer
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.weighted_reward = script_args.weighted_reward
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,   
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.shuffled_num_generations = self.num_generations // 2
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,   
            temperature=1, # HACK
            num_return_sequences=self.shuffled_num_generations,
            pad_token_id=pad_token_id,
        )
        self.dummy_generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            top_p=0.95,   
            temperature=1, # HACK
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        self.len_control = script_args.len_control
        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_per_token_logps(self, model, input_ids, **kwargs):
        logits = model.thinker(input_ids, **kwargs).logits
        # logits = model(input_ids, **kwargs).logits
        
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        prompts = [x["prompt"] for x in inputs]
        
        input_copy = copy.deepcopy(inputs[0]['prompt'])
        input_copy = self.remove_none_from_data(input_copy)

        if inputs[0]['data_type'] == 'image':
            input_copy[0]['content'][0]['image'] = "../../../datasets/AV-TAU-R1/" + inputs[0]['path'][0:]
        elif inputs[0]['data_type'] == 'video':
            input_copy[0]['content'][0]['video'] = "../../../datasets/vidi-r1/" + inputs[0]['path']['video']
        elif inputs[0]['data_type'] == 'image_audio':
            image_path = "../../../datasets/AVQA_R1/train/" + inputs[0]['path']['image']
            audio_path = "../../../datasets/AVQA_R1/train/" + inputs[0]['path']['audio']

            # 加入 content
            input_copy[0]['content'][0]['image'] = image_path
            input_copy[0]['content'][1]['audio'] = audio_path

        audios, images, videos = process_mm_info(input_copy, use_audio_in_video=self.use_audio_in_video)


        text = self.processing_class.apply_chat_template(
            input_copy,  # 对话格式数据
            add_generation_prompt=True, 
            tokenize=False
        )

        prompt_inputs = self.processing_class(
            text=text,  
            audio=audios,
            images=images, 
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video
        )
        
        
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # 为Qwen2.5Omni模型生成文本和音频
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs, 
                generation_config=self.generation_config,
                use_audio_in_video=self.use_audio_in_video,
                return_audio=False
            )
            
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            
                
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        for key in ["input_ids", "attention_mask"]:
            if key in prompt_inputs:
                prompt_inputs.pop(key)
                
        repeat_factor = len(prompt_completion_ids)

        if inputs[0]['data_type'] == 'image' or inputs[0]['data_type'] == 'image_clver':
            # 处理图像
            if "pixel_values" in prompt_inputs:
                prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(repeat_factor, 1)
                if "image_grid_thw" in prompt_inputs:
                    prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(repeat_factor, 1)


        if inputs[0]['data_type'] == 'video':
            # 处理视频
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            if 'second_per_grid_ts' in prompt_inputs:
                del prompt_inputs["second_per_grid_ts"]

            # 处理音频
            if "input_features" in prompt_inputs:
                prompt_inputs["input_features"] = prompt_inputs["input_features"].repeat_interleave(repeat_factor, dim=0)
                if "feature_attention_mask" in prompt_inputs:
                    prompt_inputs["feature_attention_mask"] = prompt_inputs["feature_attention_mask"].repeat_interleave(repeat_factor, dim=0)

        if inputs[0]['data_type'] == 'image_audio':
            # 处理图像
            if "pixel_values" in prompt_inputs:
                prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(repeat_factor, 1)
                if "image_grid_thw" in prompt_inputs:
                    prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(repeat_factor, 1)

            # 处理音频
            if "input_features" in prompt_inputs:
                prompt_inputs["input_features"] = prompt_inputs["input_features"].repeat_interleave(repeat_factor, dim=0)
                if "feature_attention_mask" in prompt_inputs:
                    prompt_inputs["feature_attention_mask"] = prompt_inputs["feature_attention_mask"].repeat_interleave(repeat_factor, dim=0)


        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]
        
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, **prompt_inputs)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # 计算KL散度
        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1
        
        # 解码生成的完成
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
        
        # 计算奖励
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # === 总奖励 START ===
        if not self.weighted_reward:
            rewards = rewards_per_func.sum(dim=1)
        else:
            # [accuracy reward, format reward]
            reward_weights = torch.tensor([2.0, 1.0], dtype=torch.float32, device=device) 
            rewards = (rewards_per_func * reward_weights).sum(dim=1)

    
        # 长度控制奖励
        if self.len_control:
            mask = rewards_per_func[:, 0] > 0.1
            lenth_list = completion_mask.sum(1)
            selected_indices = torch.nonzero(mask, as_tuple=True)[0].tolist()
                    
            if len(selected_indices) > 1:     
                for idx in selected_indices:
                    # if 320 <= lenth_list[idx] <= 512:
                    if 160 <= lenth_list[idx] <= 256:
                        rewards[idx] += 0.2

        # 分组奖励计算
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # 奖励归一化
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # 损失计算
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        
        # With KL loss
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        

        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # 记录指标
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        
        gathered_rewards = self.accelerator.gather_for_metrics(rewards)
        
        # num_devices = gathered_rewards.size(0) // self.num_generations 
        # rewards_per_device = gathered_rewards.view(num_devices, self.num_generations)
        rewards_per_device, num_devices = self.safe_group_rewards(gathered_rewards, self.num_generations, mode="pad")
        wrong_devices = (rewards_per_device <= 1).all(dim=1)
        wrong_ratio = wrong_devices.sum().item() / num_devices
        
        correct_devices = (rewards_per_device >= 2).all(dim=1)
        correct_ratio = correct_devices.sum().item() / num_devices
        
        self._metrics["all_wrong"].append(wrong_ratio)
        self._metrics["all_correct"].append(correct_ratio)
        
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

  
        return loss
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()


    def safe_group_rewards(self, gathered_rewards: torch.Tensor, num_generations: int, mode: str = "pad"):
        total_samples = gathered_rewards.shape[0]
        remainder = total_samples % num_generations

        if remainder != 0:
            if mode == "pad":
                pad_size = num_generations - remainder
                pad_values = torch.zeros(pad_size, device=gathered_rewards.device, dtype=gathered_rewards.dtype)
                gathered_rewards = torch.cat([gathered_rewards, pad_values], dim=0)
                print(f"[safe_group_rewards] Warning: padded {pad_size} samples to match num_generations")
            elif mode == "truncate":
                gathered_rewards = gathered_rewards[: total_samples - remainder]
                print(f"[safe_group_rewards] Warning: truncated {remainder} extra samples to match num_generations")
            else:
                raise ValueError(f"Unsupported mode: {mode}. Use 'pad' or 'truncate'.")

        num_devices = gathered_rewards.size(0) // num_generations
        reshaped_rewards = gathered_rewards.view(num_devices, num_generations)
        return reshaped_rewards, num_devices
