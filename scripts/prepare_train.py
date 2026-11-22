from collections import Counter
from pathlib import Path
import random

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
def read(name, columns):
    return pq.read_table(ROOT / "data/nq_hotpotqa_train" / f"{name}.parquet", columns=columns)

train = read("train", ["id", "prompt", "question", "data_source", "reward_model", "extra_info"])
questions = [q.strip().casefold() for q in train["question"].to_pylist()]
sources = train["data_source"].to_pylist()
counts = Counter(questions)
test_questions = {q.strip().casefold() for q in read("test", ["question"])["question"].to_pylist()}
rng = random.Random(42)
validation = set()
for source, size in [("nq", 478), ("hotpotqa", 546)]:
    eligible = [i for i, q in enumerate(questions) if sources[i] == source and counts[q] == 1 and q not in test_questions]
    validation.update(rng.sample(eligible, size))

data = train.append_column("agent_name", pa.array(["search_agent"] * len(train)))
folder = ROOT / "data/grpo-search"
folder.mkdir(exist_ok=True)
for name, rows in [("train", [i for i in range(len(data)) if i not in validation]), ("validation", sorted(validation))]:
    table = data.take(rows)
    pq.write_table(table, folder / f"{name}.parquet")
    print(name, len(table))
