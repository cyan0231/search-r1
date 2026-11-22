from pathlib import Path

import os

import faiss
import uvicorn
from datasets import load_dataset
from fastapi import FastAPI
from pydantic import BaseModel, Field

from index_builder import load_encoder, encode


ROOT = Path(__file__).resolve().parent.parent

CORPUS_PATH = Path(os.getenv("CORPUS_PATH", ROOT / "data/wiki18/wiki-18.jsonl"))
INDEX_PATH = Path(os.getenv("INDEX_PATH", ROOT / "indexes/wiki18-full/e5_Flat.index"))
CACHE_PATH = ROOT / ".cache/huggingface/datasets"

# 一个请求可以包含很多问题，内部每 512 条处理一次
RETRIEVAL_BATCH_SIZE = 512


class SearchRequest(BaseModel):
    queries: list[str]
    topk: int = Field(default=3, ge=1, le=20)
    return_scores: bool = True


def main():
    # 1. 服务启动时加载一次
    print("正在加载原文……", flush=True)
    documents = load_dataset(
        "json",
        data_files=str(CORPUS_PATH),
        split="train",
        cache_dir=str(CACHE_PATH),
        keep_in_memory=False,
    )

    print("正在加载索引……", flush=True)
    index = faiss.read_index(str(INDEX_PATH))
    assert index.ntotal == len(documents)
    assert index.d == 768

    print("正在加载 E5……", flush=True)
    tokenizer, model = load_encoder()

    app = FastAPI()

    # 2. 每个 HTTP 请求可以携带多个问题
    @app.post("/retrieve")
    def retrieve(request: SearchRequest):
        results = []

        # 将本次请求里的问题分批处理
        for start in range(0, len(request.queries), RETRIEVAL_BATCH_SIZE):
            queries = request.queries[
                start:start + RETRIEVAL_BATCH_SIZE
            ]

            # 一批问题 → 一批向量
            vectors = encode(
                queries, tokenizer, model, "query: "
            )

            # 一批向量 → 每个问题对应的文档位置和分数
            scores, positions = index.search(vectors, request.topk)

            # 按问题的输入顺序组织结果
            for query_positions, query_scores in zip(positions, scores):
                hits = []

                for position, score in zip(query_positions, query_scores):
                    if position < 0:
                        continue

                    document = documents[int(position)]

                    if request.return_scores:
                        hits.append({
                            "document": document,
                            "score": float(score),
                        })
                    else:
                        hits.append(document)

                results.append(hits)

        return {"result": results}

    print(f"准备完成，共 {index.ntotal:,} 条文档。", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()