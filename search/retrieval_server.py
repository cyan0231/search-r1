from pathlib import Path

import faiss
from datasets import load_dataset

from index_builder import load_encoder, encode


ROOT = Path(__file__).resolve().parent.parent

CORPUS_PATH = ROOT / "data/wiki18/wiki-18.jsonl"
INDEX_PATH = ROOT / "indexes/wiki18-full/e5_Flat.index"
CACHE_PATH = ROOT / ".cache/huggingface/datasets"

TOP_K = 3


def main():
    # 1. 加载原文
    # 第一次会生成磁盘缓存，后续可以复用。
    print("正在加载文档……", flush=True)
    documents = load_dataset(
        "json",
        data_files=str(CORPUS_PATH),
        split="train",
        cache_dir=str(CACHE_PATH),
        keep_in_memory=False,
    )

    # 2. 加载 FAISS 索引
    print("正在加载索引……", flush=True)
    index = faiss.read_index(str(INDEX_PATH))

    assert index.ntotal == len(documents), "索引和文档数量不一致"
    assert index.d == 768, "索引维度不符合预期"

    # 3. 加载 E5
    print("正在加载 E5……", flush=True)
    tokenizer, model = load_encoder()

    print(f"准备完成，共 {len(documents):,} 条文档。")

    # 4. 循环接收问题
    while True:
        question = input("\n输入问题，输入 exit 退出：").strip()

        if question == "exit":
            break
        if not question:
            continue

        # 问题 → 向量
        query_vector = encode(
            [question], tokenizer, model, "query: "
        )

        # 向量 → 最相关的文档位置和分数
        scores, positions = index.search(query_vector, TOP_K)

        # 文档位置 → 原文
        for rank, (position, score) in enumerate(
            zip(positions[0], scores[0]), start=1
        ):
            if position < 0:
                continue

            document = documents[int(position)]

            print(
                f"\n第 {rank} 名"
                f" | 相似度：{score:.4f}"
                f" | 文档 ID：{document['id']}"
            )
            print(document["contents"])


if __name__ == "__main__":
    main()