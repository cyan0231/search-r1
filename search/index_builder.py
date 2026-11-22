import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


# 文件位置：当前脚本的上两级就是项目目录
ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models/e5-base-v2"
CORPUS_PATH = ROOT / "data/wiki18/smoke/corpus.jsonl"
INDEX_PATH = ROOT / "indexes/wiki18-smoke/e5_Flat.index"


def load_encoder():
    """加载 E5 模型和分词器，只需要执行一次。"""
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        MODEL_PATH, local_files_only=True
    )
    model = model.cuda().half().eval()
    return tokenizer, model


@torch.no_grad()
def encode(texts, tokenizer, model, prefix):
    """把一批文字转换成向量。"""

    # E5 要求：文档加 passage:，问题加 query:
    texts = [prefix + text for text in texts]

    # 把文字转为模型输入；补齐长度时会产生 padding
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 模型为每个 token 生成一个向量
    output = model(**inputs)

    token_vectors = output.last_hidden_state

    # 去掉 padding 的贡献，对有效 token 的向量求平均
    mask = inputs["attention_mask"].unsqueeze(-1)
    token_vectors = token_vectors.masked_fill(~mask.bool(), 0.0)
    vectors = token_vectors.sum(dim=1) / mask.sum(dim=1)

    # 把向量长度归一化为 1
    vectors = torch.nn.functional.normalize(vectors, dim=-1)

    # FAISS 的 CPU 索引接收 float32 NumPy 数组
    return vectors.float().cpu().numpy()


def main():
    # 1. 读取文档，保留原始顺序
    with CORPUS_PATH.open(encoding="utf-8") as f:
        documents = [json.loads(line) for line in f]

    # 教学版只处理当前的小样本，避免误加载全量语料
    assert len(documents) == 2048, "这里应使用 2048 条测试语料"

    # 2. 加载 E5
    tokenizer, model = load_encoder()

    # 3. 每批编码 32 条文档
    batches = []
    for start in range(0, len(documents), 32):
        texts = [
            doc["contents"]
            for doc in documents[start:start + 32]
        ]
        vectors = encode(texts, tokenizer, model, "passage: ")
        batches.append(vectors)
        print(f"已编码 {start + len(texts)}/{len(documents)}")

    # 合并成形状为 (2048, 768) 的数组
    vectors = np.concatenate(batches, axis=0)

    # 4. 创建内积索引，并放入文档向量
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    # 5. 保存到磁盘
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    print(f"索引已保存：{INDEX_PATH}")


if __name__ == "__main__":
    main()
