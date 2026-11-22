import asyncio
import json
import re
import string
import time
from datetime import datetime
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models/Qwen2.5-3B-Instruct"
DATA_PATH = ROOT / "data/nq_hotpotqa_train/test.parquet"
OUTPUT_DIR = ROOT / "results" / datetime.now().strftime("baseline-gpu-retrieval-%Y%m%d-%H%M%S")

PORTS = [8001, 8002, 8003, 8004, 8005]
CONCURRENCY_PER_GPU = 32
SEARCH_URL = "http://127.0.0.1:8000/retrieve"
MAX_SEARCHES = 4
MAX_ROUNDS = 5
MAX_NEW_TOKENS = 512
MAX_CONTEXT_TOKENS = 8192


async def post_with_retry(client, url, **kwargs):
    # 只重试网络传输错误；服务返回 4xx/5xx 时仍直接报错。
    for attempt in range(3):
        try:
            return await client.post(url, **kwargs)
        except (httpx.NetworkError, httpx.RemoteProtocolError, httpx.TimeoutException) as error:
            if attempt == 2:
                raise
            print(
                f"HTTP 请求重试 {attempt + 1}/2：{url} | {type(error).__name__}: {error}",
                flush=True,
            )
            await asyncio.sleep(attempt + 1)


def normalize_answer(text):
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, answers):
    prediction = normalize_answer(prediction)
    return int(any(prediction == normalize_answer(answer) for answer in answers))


# 每道题把搜索词和一个等待结果的位置放进队列。
async def search(query, search_queue):
    future = asyncio.get_running_loop().create_future()
    await search_queue.put((query, future))
    return await future


# 集中发送检索请求，避免 160 道题同时逐条扫描完整 FAISS 索引。
async def retrieval_loop(search_queue, client):
    while True:
        batch = [await search_queue.get()]
        await asyncio.sleep(0.02)
        while len(batch) < 64 and not search_queue.empty():
            batch.append(search_queue.get_nowait())

        try:
            response = await post_with_retry(
                client,
                SEARCH_URL,
                json={
                    "queries": [query for query, future in batch],
                    "topk": 3,
                    "return_scores": True,
                },
                timeout=600,
            )
            response.raise_for_status()
            results = response.json()["result"]
            if len(results) != len(batch):
                raise RuntimeError("检索返回的结果数量与问题数量不一致")

            for (_, future), documents in zip(batch, results):
                passages = []
                for rank, hit in enumerate(documents, start=1):
                    title, _, body = hit["document"]["contents"].partition("\n")
                    passages.append(f"Doc {rank}(Title: {title}) {body}")
                if not future.done():
                    future.set_result("\n".join(passages))
        except Exception as error:
            # 接口故障传给正在等待的题目，让评估报错停止，不算成模型答错。
            for _, future in batch:
                if not future.done():
                    future.set_exception(error)


async def generate(context, tokenizer, client, port):
    token_ids = tokenizer.encode(context, add_special_tokens=False)
    remaining = MAX_CONTEXT_TOKENS - len(token_ids)
    if remaining <= 0:
        return None

    response = await post_with_retry(
        client,
        f"http://127.0.0.1:{port}/v1/completions",
        json={
            "model": "qwen-search",
            "prompt": token_ids,
            "max_tokens": min(MAX_NEW_TOKENS, remaining),
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "stop": ["</search>", "</answer>"],
            "include_stop_str_in_output": True,
            "stop_token_ids": [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            ],
            "add_special_tokens": False,
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["text"]


async def answer_question(messages, tokenizer, client, port, search_queue):
    context = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    searches = 0
    for _ in range(MAX_ROUNDS):
        text = await generate(context, tokenizer, client, port)
        if text is None:
            return "", searches, "context_limit"

        match = re.search(r"<(search|answer)>(.*?)</\1>", text, re.DOTALL)
        if not match:
            return "", searches, "invalid_format"

        action, content = match.group(1), match.group(2).strip()
        if not content:
            return "", searches, "empty_content"
        context += text[:match.end()]

        if action == "answer":
            return content, searches, "answered"
        if searches >= MAX_SEARCHES:
            return "", searches, "search_limit"

        passages = await search(content, search_queue)
        searches += 1
        context += f"\n\n<information>{passages}</information>\n\n"

    return "", searches, "round_limit"


# 每个 worker 固定使用一个模型服务，但从公共题目列表中领取新题。
async def worker(port, jobs, tokenizer, client, search_queue, output, stats, total):
    for index, row in jobs:
        prediction, searches, status = await answer_question(
            row["prompt"], tokenizer, client, port, search_queue,
        )
        score = exact_match(prediction, row["golden_answers"]) if status == "answered" else 0
        result = {
            "index": index,
            "id": row["id"],
            "data_source": row["data_source"],
            "question": row["question"],
            "golden_answers": row["golden_answers"],
            "prediction": prediction,
            "em": score,
            "search_count": searches,
            "status": status,
            "server_port": port,
        }
        output.write(json.dumps(result, ensure_ascii=False) + "\n")
        output.flush()

        stats["completed"] += 1
        stats["correct"] += score
        stats["searches"] += searches
        counts = stats["status_counts"]
        counts[status] = counts.get(status, 0) + 1

        source = row["data_source"]
        if source not in stats["by_data_source"]:
            stats["by_data_source"][source] = {"total": 0, "correct": 0}
        stats["by_data_source"][source]["total"] += 1
        stats["by_data_source"][source]["correct"] += score

        n = stats["completed"]
        if n % 50 == 0 or n == total:
            elapsed = time.perf_counter() - stats["started"]
            print(
                f"[{n:,}/{total:,}] EM：{stats['correct'] / n:.2%} | "
                f"平均搜索：{stats['searches'] / n:.2f} | "
                f"速度：{n / elapsed * 60:.1f} 题/分钟",
                flush=True,
            )


async def main():
    data = pq.read_table(
        DATA_PATH,
        columns=["id", "question", "prompt", "golden_answers", "data_source"],
    ).to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"测试题数：{len(data):,}，总并发：{len(PORTS) * CONCURRENCY_PER_GPU}", flush=True)
    print(f"结果目录：{OUTPUT_DIR}", flush=True)

    jobs = iter(enumerate(data))
    search_queue = asyncio.Queue()
    stats = {
        "completed": 0, "correct": 0, "searches": 0,
        "status_counts": {}, "by_data_source": {},
        "started": time.perf_counter(),
    }

    async with httpx.AsyncClient(
        trust_env=False, timeout=httpx.Timeout(600.0, connect=10.0),
        limits=httpx.Limits(
            max_connections=len(PORTS) * CONCURRENCY_PER_GPU + 8,
            max_keepalive_connections=len(PORTS) * CONCURRENCY_PER_GPU + 8,
        ),
    ) as client:
        with (OUTPUT_DIR / "predictions.jsonl").open("w", encoding="utf-8") as output:
            retrieval_task = asyncio.create_task(retrieval_loop(search_queue, client))
            workers = [
                asyncio.create_task(worker(
                    port, jobs, tokenizer, client, search_queue, output, stats, len(data),
                ))
                for port in PORTS
                for _ in range(CONCURRENCY_PER_GPU)
            ]
            try:
                await asyncio.gather(*workers)
            finally:
                retrieval_task.cancel()
                for task in workers:
                    task.cancel()
                await asyncio.gather(retrieval_task, *workers, return_exceptions=True)

    for values in stats["by_data_source"].values():
        values["em"] = values["correct"] / values["total"]

    summary = {
        "model": str(MODEL_PATH), "dataset": str(DATA_PATH),
        "settings": {
            "backend": "vllm", "dtype": "float16",
            "prompt": "测试集自带的作者原版 prompt",
            "ports": PORTS, "concurrency_per_gpu": CONCURRENCY_PER_GPU,
            "temperature": 0.0, "max_searches": MAX_SEARCHES,
            "max_rounds": MAX_ROUNDS, "max_new_tokens_per_round": MAX_NEW_TOKENS,
            "max_context_tokens": MAX_CONTEXT_TOKENS, "retrieval_topk": 3,
            "retrieval_batch_max": 64, "retrieval_batch_wait_seconds": 0.02,
            "retrieval_backend": "faiss_gpu_flat_ip",
            "http_max_attempts": 3,
            "retrieval_index_storage_dtype": "float16",
            "retrieval_index": str(ROOT / "indexes/wiki18-full/e5_Flat.index"),
            "retrieval_encoder": str(ROOT / "models/e5-base-v2"),
        },
        "total": stats["completed"], "correct": stats["correct"],
        "em": stats["correct"] / stats["completed"],
        "average_searches": stats["searches"] / stats["completed"],
        "elapsed_seconds": time.perf_counter() - stats["started"],
        "status_counts": stats["status_counts"],
        "by_data_source": stats["by_data_source"],
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
