import json
import re
import string
import sys
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
## 这是单卡单题串行的脚本 速度很慢

ROOT = Path(__file__).resolve().parent.parent

# 复用 search 文件夹里的 HTTP 客户端
sys.path.insert(0, str(ROOT / "search"))
from search_client import search


MODEL_PATH = ROOT / "models/Qwen2.5-3B-Instruct"
DATA_PATH = ROOT / "data/nq_hotpotqa_train/test.parquet"
OUTPUT_DIR = ROOT / "results" / datetime.now().strftime("baseline-%Y%m%d-%H%M%S")

MAX_SEARCHES = 4
MAX_ROUNDS = 5
MAX_NEW_TOKENS = 512
MAX_CONTEXT_TOKENS = 8192


# 与仓库 EM 评分采用相同的答案归一化规则
def normalize_answer(text):
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, golden_answers):
    prediction = normalize_answer(prediction)
    return int(any(
        prediction == normalize_answer(answer)
        for answer in golden_answers
    ))


@torch.no_grad()
def generate(context, tokenizer, model):
    inputs = tokenizer(
        context,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")

    input_length = inputs["input_ids"].shape[1]
    remaining = MAX_CONTEXT_TOKENS - input_length

    if remaining <= 0:
        return None

    output = model.generate(
        **inputs,
        max_new_tokens=min(MAX_NEW_TOKENS, remaining),
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
        stop_strings=["</search>", "</answer>"],
        tokenizer=tokenizer,
    )

    new_tokens = output[0, input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# 一道题的多轮搜索过程；不接收标准答案
def answer_question(messages, tokenizer, model, session):
    context = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    search_count = 0

    for _ in range(MAX_ROUNDS):
        text = generate(context, tokenizer, model)

        if text is None:
            return "", search_count, "context_limit"

        match = re.search(
            r"<(search|answer)>(.*?)</\1>",
            text,
            flags=re.DOTALL,
        )

        if not match:
            return "", search_count, "invalid_format"

        action = match.group(1)
        content = match.group(2).strip()

        if not content:
            return "", search_count, "empty_content"

        context += text[:match.end()]

        if action == "answer":
            return content, search_count, "answered"

        if search_count >= MAX_SEARCHES:
            return "", search_count, "search_limit"

        passages = search(content, session)
        search_count += 1
        context += f"\n\n<information>{passages}</information>\n\n"

    return "", search_count, "round_limit"


def main():
    # 1. 读取测试集
    data = pq.read_table(
        DATA_PATH,
        columns=["id", "question", "prompt", "golden_answers", "data_source"],
    ).to_pylist()

    print(f"测试题数：{len(data):,}", flush=True)

    # 2. 加载模型一次
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).cuda().eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"结果目录：{OUTPUT_DIR}", flush=True)

    correct = 0
    total_searches = 0
    source_stats = {}
    status_counts = {}

    # 3. 逐题生成答案，每完成一题就保存
    with requests.Session() as session:
        session.trust_env = False

        with (OUTPUT_DIR / "predictions.jsonl").open(
            "w", encoding="utf-8"
        ) as output_file:
            for number, row in enumerate(data, start=1):
                prediction, search_count, status = answer_question(
                    row["prompt"], tokenizer, model, session
                )

                # 生成结束后才使用标准答案评分
                score = (
                    exact_match(prediction, row["golden_answers"])
                    if status == "answered"
                    else 0
                )

                result = {
                    "index": number - 1,
                    "id": row["id"],
                    "data_source": row["data_source"],
                    "question": row["question"],
                    "golden_answers": row["golden_answers"],
                    "prediction": prediction,
                    "em": score,
                    "search_count": search_count,
                    "status": status,
                }
                output_file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                output_file.flush()

                correct += score
                total_searches += search_count
                status_counts[status] = status_counts.get(status, 0) + 1

                source = row["data_source"]
                if source not in source_stats:
                    source_stats[source] = {"total": 0, "correct": 0}
                source_stats[source]["total"] += 1
                source_stats[source]["correct"] += score

                if number % 10 == 0 or number == len(data):
                    print(
                        f"[{number:,}/{len(data):,}] "
                        f"EM：{correct / number:.2%} | "
                        f"平均搜索次数：{total_searches / number:.2f}",
                        flush=True,
                    )

    # 4. 汇总总体结果，以及每个数据集的结果
    for stats in source_stats.values():
        stats["em"] = stats["correct"] / stats["total"]

    summary = {
        "model": str(MODEL_PATH),
        "dataset": str(DATA_PATH),
        "settings": {
            "prompt": "测试集自带的作者原版 prompt",
            "do_sample": False,
            "dtype": "float16",
            "max_searches": MAX_SEARCHES,
            "max_rounds": MAX_ROUNDS,
            "max_new_tokens_per_round": MAX_NEW_TOKENS,
            "max_context_tokens": MAX_CONTEXT_TOKENS,
            "retrieval_topk": 3,
        },
        "total": len(data),
        "correct": correct,
        "em": correct / len(data),
        "average_searches": total_searches / len(data),
        "status_counts": status_counts,
        "by_data_source": source_stats,
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()