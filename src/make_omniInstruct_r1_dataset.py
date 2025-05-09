import os
import json
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

from PIL import UnidentifiedImageError

# === 配置 ===
split_name = "train"  # 可改为 "valid"
dataset = load_dataset("m-a-p/OmniInstruct_v1", split=split_name)
output_dir = f"../datasets/OmniInstruct_V1_AVQA_R1/{split_name}"
json_name = f"omni_rl_format_{split_name}.json"
audio_dir = os.path.join(output_dir, "audios")
image_dir = os.path.join(output_dir, "images")

# === 创建文件夹 ===
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

print(f"[INFO] Processing split: {split_name}")
print(f"[INFO] Saving to: {os.path.join(output_dir, json_name)}")

final_data = []

for idx, example in tqdm(enumerate(dataset), total=len(dataset)):
    try:
        if example.get("source", "") != "AVQA":
            continue  # 跳过非 AVQA 的样本
        # === 提取音频 ===
        audio_array = example["audio"]["array"]
        audio_rate = example["audio"]["sampling_rate"]
        audio_rel_path = f"audios/sample_{idx}.wav"
        audio_full_path = os.path.join(output_dir, audio_rel_path)

        # === 提取图像 ===
        image_rel_path = f"images/sample_{idx}.jpg"
        image_full_path = os.path.join(output_dir, image_rel_path)

        # 提前尝试保存，验证合法性（尤其图像）
        sf.write(audio_full_path, audio_array, audio_rate)
        example["image"].save(image_full_path)

        # === 获取正确答案 ===
        if example["answer"] not in example["options"]:
            raise ValueError("Answer not in options")

        correct_idx = example["options"].index(example["answer"])
        correct_label = "ABCD"[correct_idx]

        # === 构建输出项 ===
        new_item = {
            "problem_id": idx,
            "problem": example["question"],
            "data_type": "image_audio",
            "problem_type": "multiple choice",
            "options": [
                f"{l}. {t}" for l, t in zip("ABCD", example["options"])
            ],
            "solution": f"<answer>{correct_label}</answer>",
            "path": {
                "image": image_rel_path,
                "audio": audio_rel_path
            },
            "data_source": "OmniInstruct_v1-"+ example["source"]
        }

        final_data.append(new_item)
        # break

    except (KeyError, ValueError, UnidentifiedImageError, Exception) as e:
        print(f"[Warning] Skipped index {idx} due to error: {e}")
        # 如果有文件已保存，删掉
        if os.path.exists(audio_full_path):
            os.remove(audio_full_path)
        if os.path.exists(image_full_path):
            os.remove(image_full_path)
        print(example["source"], example["answer"], example["options"])
        continue

# === 保存最终 JSON ===
with open(os.path.join(output_dir, json_name), "w") as f:
    json.dump(final_data, f, indent=2)

print(f"[INFO] Finished. Valid samples: {len(final_data)}")
