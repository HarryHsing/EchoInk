import json
from collections import defaultdict

RESULT_PATH = "./exp_results/omniBench_Qwen2.5-Omni-7B_results.jsonl"

# 初始化统计容器
overall = {"total": 0, "correct": 0}
by_task = defaultdict(lambda: {"total": 0, "correct": 0})
by_audio = defaultdict(lambda: {"total": 0, "correct": 0})
by_variant = defaultdict(lambda: {"total": 0, "correct": 0})

with open(RESULT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        gt, pred = item["gt"], item["pred"]
        task = item.get("task_type", "unknown")
        audio = item.get("audio_type", "unknown")
        variant = item.get("variant", "unknown")

        # 总体统计
        overall["total"] += 1
        overall["correct"] += int(pred == gt)

        # 按任务类型
        by_task[task]["total"] += 1
        by_task[task]["correct"] += int(pred == gt)

        # 按音频类型
        by_audio[audio]["total"] += 1
        by_audio[audio]["correct"] += int(pred == gt)

        # 按模型版本（base / rl）
        by_variant[variant]["total"] += 1
        by_variant[variant]["correct"] += int(pred == gt)

# 打印结果
def show_stats(title, stats_dict):
    print(f"\n📊 {title}")
    for k, v in sorted(stats_dict.items()):
        total = v["total"]
        acc = v["correct"] / total if total > 0 else 0
        print(f"- {k:25} | Acc: {acc:.2%} | Total: {total: d}")


print(f"\n✅ Overall accuracy: {overall['correct'] / overall['total']:.2%}")
show_stats("By Task Type", by_task)
show_stats("By Audio Type", by_audio)
show_stats("By Variant", by_variant)
