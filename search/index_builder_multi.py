"""多卡 E5 建库：每卡一个进程，动态领取语料块，最后按原文顺序合并。

只需要现有检索环境；不依赖同目录的 index_builder.py。
路径相对于项目根目录，--gpus 指定参与建库的 GPU UUID。
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import time
import traceback


DIM = 768


def read_range(path, begin, end):
    """只读取起始字节位于 [begin, end) 的完整 JSONL 行，不重复、不漏行。"""
    with open(path, "rb") as f:
        if begin:
            f.seek(begin - 1)
            if f.read(1) != b"\n":
                f.readline()  # 边界落在一行中间：这行归前一个块。
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            doc = json.loads(line)
            if not isinstance(doc["contents"], str):
                raise TypeError("contents 必须是字符串")
            yield doc["contents"]


def prefetch(iterator):
    """GPU 处理当前批时，后台线程准备下一批分词结果；最多预取一批。"""
    iterator = iter(iterator)
    sentinel = object()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(next, iterator, sentinel)
        while True:
            item = future.result()
            if item is sentinel:
                return
            future = pool.submit(next, iterator, sentinel)
            yield item


def worker(slot, gpu, args, tasks, results):
    # 必须在导入 torch、初始化 CUDA 之前绑定 GPU。
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(args.cpu_threads)
    try:
        import faiss
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(args.cpu_threads)
        faiss.omp_set_num_threads(args.cpu_threads)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        model = AutoModel.from_pretrained(args.model, local_files_only=True)
        model = model.cuda().half().eval()
        print(f"[卡{slot}] {torch.cuda.get_device_name(0)}，模型已加载", flush=True)

        def tokenize(texts):
            return tokenizer(
                ["passage: " + text for text in texts],
                padding=True, truncation=True, max_length=256,
                return_tensors="pt",
            )

        @torch.no_grad()
        def encode_inputs(inputs):
            # 保留调用方的 CPU 输入，必要时能缩小批次重试。
            inputs = {name: value.to("cuda") for name, value in inputs.items()}
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            hidden = hidden.masked_fill(~mask.bool(), 0.0)
            vectors = hidden.sum(dim=1) / mask.sum(dim=1)
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
            return vectors.float().cpu().numpy()

        def prepared_batches(texts, batch_size):
            texts = iter(texts)
            while True:
                batch = list(itertools.islice(texts, batch_size))
                if not batch:
                    return
                yield tokenize(batch)

        # 所有卡用相同样本，各自比较 batch 的实际吞吐量。
        # 预热一次；两轮计时取平均；显存不足的候选不采用。
        sample = list(itertools.islice(
            read_range(args.corpus, 0, os.path.getsize(args.corpus)), 2048
        ))
        if not sample:
            raise ValueError("语料为空")
        best_batch, best_speed = None, 0.0
        candidates = [args.batch_size] if args.batch_size else [64, 128, 256, 512, 1024]
        for batch_size in candidates:
            oom = False
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                encode_inputs(tokenize(sample[:batch_size]))
                torch.cuda.synchronize()
                started = time.perf_counter()
                for _ in range(2):
                    for inputs in prefetch(prepared_batches(sample, batch_size)):
                        encode_inputs(inputs)
                torch.cuda.synchronize()
                speed = 2 * len(sample) / (time.perf_counter() - started)
                peak = torch.cuda.max_memory_allocated() / 1024**3
                print(f"[卡{slot}] batch={batch_size}: {speed:.0f} 条/秒，"
                      f"张量峰值显存 {peak:.2f} GiB", flush=True)
                if speed > best_speed:
                    best_batch, best_speed = batch_size, speed
            except torch.cuda.OutOfMemoryError:
                oom = True
                print(f"[卡{slot}] batch={batch_size} 显存不足，停止增大", flush=True)
            if oom:
                torch.cuda.empty_cache()
                break
        if best_batch is None:
            raise RuntimeError("候选 batch 均无法运行，请减少 batch 或检查 GPU 占用")
        del sample
        torch.cuda.empty_cache()
        print(f"[卡{slot}] 选择 batch={best_batch}，开始领取任务", flush=True)

        completed = 0
        started = time.perf_counter()
        while True:
            task = tasks.get()
            if task is None:
                return
            part, begin, end = task
            index = faiss.IndexFlatIP(DIM)
            texts = read_range(args.corpus, begin, end)
            batches = prefetch(prepared_batches(texts, best_batch))
            for inputs in batches:
                offset = 0
                while offset < len(inputs["input_ids"]):
                    end_of_batch = min(offset + best_batch, len(inputs["input_ids"]))
                    oom = False
                    try:
                        vectors = encode_inputs({
                            name: value[offset:end_of_batch]
                            for name, value in inputs.items()
                        })
                    except torch.cuda.OutOfMemoryError:
                        if best_batch == 1:
                            raise
                        oom = True
                    if oom:
                        best_batch = max(1, best_batch // 2)
                        torch.cuda.empty_cache()
                        print(f"[卡{slot}] 实际文档显存不足，降到 batch={best_batch}", flush=True)
                        continue
                    if not np.isfinite(vectors).all():
                        raise RuntimeError("发现 NaN/Inf 向量")
                    index.add(vectors)
                    offset = end_of_batch
            path = Path(args.parts_dir) / f"{part:06d}.index"
            faiss.write_index(index, str(path))
            count = index.ntotal
            completed += count
            del index
            results.put((part, count))
            speed = completed / (time.perf_counter() - started)
            print(f"[卡{slot}] 完成块 {part}，累计 {completed:,} 条，"
                  f"平均 {speed:.0f} 条/秒", flush=True)
    except BaseException:
        results.put(("error", f"卡{slot}:\n{traceback.format_exc()}"))
        raise


def merge_indexes(paths, counts, output):
    """按原文块顺序追加索引，确保 FAISS 位置对应 corpus.jsonl 行号。"""
    import faiss

    merged = faiss.IndexFlatIP(DIM)
    for path, count in zip(paths, counts):
        part = faiss.read_index(str(path))
        if part.d != DIM or part.ntotal != count:
            raise RuntimeError(f"分片索引不符合预期：{path}")
        merged.merge_from(part, 0)
        del part
        print(f"[合并] 已加入 {merged.ntotal:,} 条", flush=True)
    if merged.ntotal != sum(counts):
        raise RuntimeError("合并后的总数不正确")
    temporary = str(output) + ".tmp"
    faiss.write_index(merged, temporary)
    os.replace(temporary, output)


def check_gpus(gpus):
    """避免与旧单卡进程或其他用户的计算任务争抢 GPU。"""
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(set(gpus)) != len(gpus) or any(g not in inventory for g in gpus):
        raise RuntimeError("GPU UUID 重复或不存在，请检查 --gpus")
    running = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
         "--format=csv,noheader,nounits"], text=True
    )
    busy = [line for line in running.splitlines()
            if line.split(",")[0].strip() in gpus]
    if busy:
        raise RuntimeError("以下 GPU 已有计算进程；停止自己的旧任务，或排除占用卡：\n"
                           + "\n".join(busy))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--corpus", type=Path, default=root / "data/wiki18/wiki-18.jsonl")
    parser.add_argument("--model", type=Path, default=root / "models/e5-base-v2")
    parser.add_argument("--output", type=Path, default=root / "indexes/wiki18-full/e5_Flat.index")
    parser.add_argument("--gpus", nargs="+", required=True, help="要使用的 GPU UUID")
    parser.add_argument("--batch-size", type=int, default=0, help="0=每卡自动测速选择")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--chunk-mib", type=int, default=64, help="每块语料的字节大小")
    args = parser.parse_args()
    if args.batch_size < 0 or args.cpu_threads < 1 or args.chunk_mib < 1:
        parser.error("参数应为正数；batch-size 允许为 0")
    size = args.corpus.stat().st_size
    if not size:
        raise ValueError("语料为空")
    check_gpus(args.gpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.parts_dir = tempfile.mkdtemp(prefix="parts-", dir=args.output.parent)

    # 不实际切割语料文件；每个任务只记录一个字节区间。
    chunk_bytes = args.chunk_mib * 1024**2
    ranges = [(i, begin, min(begin + chunk_bytes, size))
              for i, begin in enumerate(range(0, size, chunk_bytes))]
    ctx = mp.get_context("spawn")
    tasks, results = ctx.Queue(), ctx.Queue()
    for task in ranges:
        tasks.put(task)
    for _ in args.gpus:
        tasks.put(None)
    workers = [ctx.Process(target=worker, args=(slot, gpu, args, tasks, results))
               for slot, gpu in enumerate(args.gpus)]
    counts = {}
    started = time.perf_counter()
    print(f"启动 {len(workers)} 张 GPU，动态分配 {len(ranges)} 个语料块。", flush=True)
    try:
        for process in workers:
            process.start()
        while len(counts) < len(ranges):
            try:
                part, count = results.get(timeout=2)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in workers):
                    raise RuntimeError("编码子进程异常退出，请查看上方日志")
                if all(p.exitcode is not None for p in workers):
                    raise RuntimeError("所有进程已退出，但结果不完整")
                continue
            if part == "error":
                raise RuntimeError(count)
            counts[part] = count
            elapsed = time.perf_counter() - started
            print(f"[总进度] {len(counts)}/{len(ranges)} 块，"
                  f"已落盘 {sum(counts.values()):,} 条，"
                  f"累计平均 {sum(counts.values()) / elapsed:.0f} 条/秒", flush=True)
        for process in workers:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError("子进程退出失败")
    finally:
        for process in workers:
            if process.is_alive():
                process.terminate()
        for process in workers:
            if process.pid is not None:
                process.join()
        tasks.cancel_join_thread()
        tasks.close()
        results.close()

    paths = [Path(args.parts_dir) / f"{i:06d}.index" for i in range(len(ranges))]
    merge_indexes(paths, [counts[i] for i in range(len(ranges))], args.output)
    shutil.rmtree(args.parts_dir)  # 正式索引写完后，删除本次临时分片。
    print(f"完成，共 {sum(counts.values()):,} 条：{args.output}", flush=True)


if __name__ == "__main__":
    main()
