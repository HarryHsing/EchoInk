#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate Qwen2-5-Omni on OmniBench_R1 (image+audio 多选).
支持断点续推推理。
"""

import os, json, time, re, torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import logging
logging.getLogger().setLevel(logging.ERROR)

# ==== 基本路径 ====
DATA_ROOT  = "../datasets/OmniInstruct_V1_AVQA_R1/valid"
JSON_PATH  = os.path.join(DATA_ROOT, "omni_rl_format_valid.json")
MODEL_PATH = "./src/r1-v/log/Qwen2.5-Omni-7B-GRPO-OmniBench-BS8-KL0.001-CommonFormat-V4-NumG-16"
OUT_PATH   = "./exp_results/AVQA-r1_Qwen2.5-Omni-7B-GRPO-BS8-KL0.001-CommonFormat-V4-NumG-16-TRAIN_CONFIG.jsonl"
USE_AUDIO  = True
# Base = True for Qwen-2.5-Omni model
# Base = False for Qwen-2.5-Omni-7B-GRPO model
BASE_MODEL = False

# ==== Prompt模板 ====
if not BASE_MODEL:
    # Common format 
    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Make sure to carefully consider both the visual and audio information before answering. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )

else:
    QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please answer the following question based on the given image and audio."
    )

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
}

def extract_choice(reply: str) -> str:
    """
    从 assistant 回复中提取预测选项。
    优先提取 <answer>X</answer>，如果没有，再从 assistant 生成内容中找 A-D。
    """
    try:
        # 找到 'assistant' 开头
        if "\nassistant" in reply:
            reply = reply.split("\nassistant", 1)[1].strip()

        # 优先匹配 <answer>X</answer>
        m = re.search(r"<answer>\s*([A-D])\s*</answer>", reply, re.I)
        if m:
            return m.group(1).upper()

        # fallback：找孤立的 A-D 字母
        m = re.search(r"\b([A-D])\b", reply, re.I)
        if m:
            return m.group(1).upper()
        
        return "?"  # 找不到
    except Exception as e:
        print(f"[extract_choice error]: {e}")
        return "?"

GT_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.I)

# ==== 读取数据 ====
with open(JSON_PATH, "r", encoding="utf-8") as f:
    samples = json.load(f)
print(f"📚 {len(samples)} samples loaded.")

# ==== 加载模型 ====
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
    enable_audio_output=False,
)
model.disable_talker()
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
print("✅ Model ready.")

# ==== 推理循环（支持断点续推） ====
correct = 0
start = time.time()

# ---- 检查已有推理结果 ----
existing_ids = set()
if os.path.exists(OUT_PATH):
    print(f"⚡️ Found existing result file {OUT_PATH}. Resuming...")
    with open(OUT_PATH, "r", encoding="utf-8") as f_exist:
        for line in f_exist:
            try:
                data = json.loads(line)
                existing_ids.add(data["id"])
            except:
                continue
    print(f"⚡️ {len(existing_ids)} samples already completed.")
else:
    print(f"🆕 No existing result file. Start fresh.")

# ---- 追加模式打开文件 ----
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "a", encoding="utf-8") as fout:
    for idx, ex in enumerate(samples):
        if ex["problem_id"] in existing_ids:
            continue  # 跳过已完成

        img_path = os.path.join(DATA_ROOT, ex["path"]["image"])
        aud_path = os.path.join(DATA_ROOT, ex["path"]["audio"])

        # --- 组装prompt ---
        question_text = ex["problem"] + "\n" + "\n".join(ex["options"])
        prompt = QUESTION_TEMPLATE.format(Question=question_text) + TYPE_TEMPLATE["multiple choice"]

        if BASE_MODEL:
            conv = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [
                    {"type": "image", "image": img_path},
                    {"type": "audio", "audio": aud_path},
                    {"type": "text", "text": prompt},
                ]}
            ]
        else:
            conv = [
                {"role": "user", "content": [
                    {"type": "image", "image": img_path},
                    {"type": "audio", "audio": aud_path},
                    {"type": "text", "text": prompt},
                ]}
            ]

        # --- 输入处理 ---
        prompt_text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, _ = process_mm_info(conv, use_audio_in_video=USE_AUDIO)
        inputs = processor(
            text=prompt_text, images=images, audio=audios,
            return_tensors="pt", padding=True,
            use_audio_in_video=USE_AUDIO
        ).to(model.device)

        # --- 生成 ---
        with torch.no_grad():
            # Inference Config
            # out_ids = model.generate(
            #     **inputs,
            #     use_audio_in_video=USE_AUDIO,
            #     return_audio=False,
            #     thinker_do_sample=False,
            #     repetition_penalty=1.0
            # )

            # Training Config
            out_ids = model.generate(
                **inputs, 
                use_audio_in_video=USE_AUDIO,
                return_audio=False,
                do_sample=True,
                top_p=0.95,   
                temperature=1, # HACK
            )
        reply = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

        # --- 评估 ---
        pred = extract_choice(reply)
        gt   = GT_RE.search(ex["solution"]).group(1)
        correct += (pred == gt)

        fout.write(json.dumps({
            "id": ex["problem_id"],
            "gt": gt,
            "pred": pred,
            "task_type": ex.get("task_type", ""),
            "audio_type": ex.get("audio_type", ""),
            "reply": reply,
            "variant": "base" if BASE_MODEL else "rl",
            # "current_correct": correct
        }, ensure_ascii=False) + "\n")
        fout.flush()

        if (idx + 1) % 50 == 0:
            print(f"[{idx+1}/{len(samples)}] Acc {correct/(idx+1):.2%}")

# ==== 总结 ====
acc = correct / len(samples)
print(f"\n🎯 {'BASE' if BASE_MODEL else 'RL'} accuracy {acc:.2%} | elapsed {(time.time()-start)/60:.1f} min")
print("Results saved to", OUT_PATH)