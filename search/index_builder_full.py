import json
from pathlib import Path

import faiss

from index_builder import load_encoder, encode


ROOT = Path(__file__).resolve().parent.parent

CORPUS_PATH = ROOT / "data/wiki18/wiki-18.jsonl"
INDEX_PATH = ROOT / "indexes/wiki18-full/e5_Flat.index"

BATCH_SIZE = 256
import time


def main():
    # 1. 加载 E5
    tokenizer, model = load_encoder()

    # 2. 创建空索引，每个向量有 768 维
    index = faiss.IndexFlatIP(768)

    texts = []
    start_time = time.perf_counter()
    # 3. 逐行读取文档，每攒够 32 条就处理
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            document = json.loads(line)
            texts.append(document["contents"])

            if len(texts) == BATCH_SIZE:
                vectors = encode(texts, tokenizer, model, "passage: ")
                index.add(vectors)
                texts = []

                if index.ntotal % 4096 == 0:
                    elapsed = time.perf_counter() - start_time
                    speed = index.ntotal / elapsed
                    print(
                        f"已处理 {index.ntotal:,} 条 | 平均 {speed:.1f} 条/秒",
                        flush=True,
                    )

    # 4. 处理最后不足 32 条的文档
    if texts:
        vectors = encode(texts, tokenizer, model, "passage: ")
        index.add(vectors)

    # 5. 所有文档处理完成，保存索引
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    print(f"完成，共 {index.ntotal:,} 条")
    print(f"索引已保存：{INDEX_PATH}")


if __name__ == "__main__":
    main()