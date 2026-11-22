from pathlib import Path

import torch
import faiss
import uvicorn
from datasets import load_dataset
from fastapi import FastAPI
from pydantic import BaseModel, Field

from index_builder import load_encoder, encode


ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data/wiki18/wiki-18.jsonl"
INDEX_PATH = ROOT / "indexes/wiki18-full/e5_Flat.index"


class SearchRequest(BaseModel):
    queries: list[str]
    topk: int = Field(default=3, ge=1, le=20)
    return_scores: bool = True


def main():
    print("读取磁盘上的完整索引……", flush=True)
    cpu_index = faiss.read_index(str(INDEX_PATH))

    resources = faiss.StandardGpuResources()
    resources.setTempMemory(1024 * 1024 * 1024)
    config = faiss.GpuIndexFlatConfig()
    config.device = 0  # 启动命令只暴露物理 1 号卡，所以进程内编号为 0。
    config.useFloat16 = True
    index = faiss.GpuIndexFlatIP(resources, cpu_index.d, config)

    # 直接查看 CPU 索引里的向量，不额外复制一份 60 GiB 数组。
    vectors = faiss.rev_swig_ptr(
        cpu_index.get_xb(), cpu_index.ntotal * cpu_index.d,
    ).reshape(cpu_index.ntotal, cpu_index.d)

    print(f"将 {cpu_index.ntotal:,} 条向量加载到 GPU，使用 FP16……", flush=True)
    # 空索引一次 add：FAISS 先预留最终空间，再在内部逐批传输和转换。
    index.add(vectors)
    torch.cuda.synchronize()
    del vectors, cpu_index
    print(f"GPU 索引加载完成：{index.ntotal:,} 条", flush=True)

    print("加载原文和 E5 模型……", flush=True)
    documents = load_dataset(
        "json",
        data_files=str(CORPUS_PATH),
        split="train",
        cache_dir=str(ROOT / ".cache/huggingface/datasets"),
        keep_in_memory=False,
    )
    tokenizer, model = load_encoder()

    app = FastAPI()

    # async 路由内同步计算：单进程中依次处理批次，避免多个线程
    # 同时使用同一个 FAISS GPU resources。客户端仍可并发提交请求。
    @app.post("/retrieve")
    async def retrieve(request: SearchRequest):
        results = []
        for start in range(0, len(request.queries), 64):
            queries = request.queries[start:start + 64]
            query_vectors = encode(queries, tokenizer, model, "query: ")
            scores, positions = index.search(query_vectors, request.topk)

            for query_positions, query_scores in zip(positions, scores):
                hits = []
                for position, score in zip(query_positions, query_scores):
                    if position < 0:
                        continue
                    document = documents[int(position)]
                    if request.return_scores:
                        hits.append({"document": document, "score": float(score)})
                    else:
                        hits.append(document)
                results.append(hits)

        return {"result": results}

    uvicorn.run(app, host="127.0.0.1", port=8000, access_log=False)


if __name__ == "__main__":
    main()
