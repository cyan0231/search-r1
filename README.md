# 用 veRL 训练 Search-R1 多轮搜索

这个实验用 Qwen2.5-3B-Instruct 学习何时搜索、怎样根据检索结果继续生成，以及何时提交答案。第 720 步模型在 51,713 道测试题上的精确匹配率（EM）为 **42.18%**。训练前用相同题目、提示词、检索工具和生成预算测得 **28.81%**，训练后多答对 6,910 题。[七个数据集的完整对比](results/test-grpo-step720-20261122/comparison.md)

## 按这四步读代码

| 阶段 | 从哪里开始 | 观察什么 |
|---|---|---|
| 建知识库 | [小样本建索引](search/index_builder.py) · [多卡全量建库](search/index_builder_multi.py) | 给文档加 `passage: ` 前缀，用 E5 做均值池化和归一化，写入 768 维 FAISS 内积索引。最终索引含 21,015,324 条段落，行号用于回查原文。 |
| 跑通一题 | [单题 demo](search/search_r1_demo.py) · [HTTP 客户端](search/search_client.py) · [检索服务](search/retrieval_http_gpu.py) | 第一次生成前套聊天模板。解析到 `<search>` 就发送查询，把返回内容包进 `<information>` 追加到原上下文。解析到 `<answer>` 就结束。 |
| 进入训练 | [SearchAgentLoop](runs/grpo-20261120-144806/verl/experimental/agent_loop/search_agent_loop.py) · [奖励函数](scripts/em_reward.py) · [正式配置](runs/grpo-20261120-144806/train.sh) | 每个问题采样四条轨迹。模型输出和工具结果都进入上下文，只有模型生成的 token 参与策略损失。 |
| 评测与选模 | [验证选模与恢复](runs/grpo-20261120-144806/run_checkpoints.py) · [评测脚本](scripts/eval_baseline_multi.py) · [训练后评测入口](results/test-grpo-step720-20261122/run.sh) | 验证 EM 选 checkpoint。测试时每题跑一条轨迹，两次全量评测用同一份评分代码。 |

如果想从最简单的实现开始读，可以先看[单卡全量建库](search/index_builder_full.py)，再用[命令行直接检索](search/retrieval_server.py)追踪“问题 → 向量 → 文档”的过程。[CPU FAISS HTTP 服务](search/retrieval_http.py)把同一查询过程封装为接口。[单卡逐题评测](scripts/eval_baseline_one-problem-per-gpu.py)展示一道题从生成、检索到 EM 评分的完整调用顺序。

### 一条样本如何变成轨迹

训练集中 `train_0` 来自 NQ，题目是 “total number of death row inmates in the us?”。`prompt` 包含指令和这道题，`reward_model.ground_truth.target` 为 `["2,718"]`，评分时读取。数据准备脚本给每条题目加上 `agent_name="search_agent"`，将请求路由到自定义循环。[数据准备代码](scripts/prepare_train.py) · [Agent 注册](configs/search_agent.yaml)

下面是单题 demo 的一次实际运行，检索返回的三段文档只摘录了与答案相关的词句。每一行按顺序追加到同一个上下文。

| 来源 | 内容 | response_mask |
|---|---|---|
| 用户提示 | Mike Barnett 代理的哪位球员后来成为 CSKA Moscow 的总经理？（题意） | prompt 单独保存 |
| 模型，第 1 轮 | `<search>Mike Barnett negotiated contracts for which NHL player who later became CSKA Moscow GM</search>` | 1 |
| 检索服务 | `<information>` 中的 Doc 1 列出了多名球员，包括 Sergei Fedorov；Doc 2 也提到了合同。 | 0 |
| 模型，第 2 轮 | `<answer>Sergei Fedorov</answer>` | 1 |

GRPO 对同题的四个奖励做组内标准化。单条奖励只取 0 或 1。预测和参考答案统一小写、删除 ASCII 标点及英文冠词、合并空白后，匹配任一参考答案即得 1 分。

## 环境和数据准备

下列命令在 Linux + NVIDIA GPU 上从仓库根目录运行。正式运行使用 Python 3.10、PyTorch 2.6.0（CUDA 12.4）、Transformers 4.51.3、Ray 2.46.0、vLLM 0.8.5.post1、flash-attn 2.7.4.post1。检索环境使用 faiss-gpu-cu12 1.8.0.2、datasets 3.6.0。`python` 指已激活的训练环境。在每个要启动检索的终端中，设置 `RETRIEVAL_PYTHON` 为装有 GPU FAISS 的 Python 可执行文件。

固定 veRL 版本，将正式运行的三个适配文件覆盖到该 checkout。训练启动脚本会将这份源码加入 `PYTHONPATH`。vLLM 适配让每轮的 512-token 限制实际生效，trainer 适配负责按验证成绩选模及保留恢复点。

~~~bash
mkdir -p vendor tmp
git clone https://github.com/verl-project/verl.git vendor/verl
git -C vendor/verl checkout --detach 8fdc4d3f202f41461f4de9f42a637228e342668b
cp -R runs/grpo-20261120-144806/verl/. vendor/verl/verl/
python -m pip install --no-deps -e vendor/verl
export RETRIEVAL_PYTHON="$(conda run -n search-r1-retrieval-gpu python -c 'import sys; print(sys.executable)')"
~~~

下载实验使用的语料、问答数据和两个模型。命令使用 Hugging Face Hub 0.30.x 提供的 CLI。Wiki 压缩包内含一层 tar 归档。

~~~bash
mkdir -p data/wiki18/smoke data/nq_hotpotqa_train models
huggingface-cli download PeterJinGo/wiki-18-corpus wiki-18.jsonl.gz --repo-type dataset --revision 69c1c00ffe7c5554c68d8548355cb22e46aabc51 --local-dir data/wiki18
tar -xzOf data/wiki18/wiki-18.jsonl.gz data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl >data/wiki18/wiki-18.jsonl
huggingface-cli download PeterJinGo/nq_hotpotqa_train train.parquet test.parquet --repo-type dataset --revision b7d80abfee334a7a91cb377544f09180d58b34f6 --local-dir data/nq_hotpotqa_train
huggingface-cli download intfloat/e5-base-v2 --local-dir models/e5-base-v2
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct
python scripts/prepare_train.py
~~~

## 从小索引到训练

**小样本检索。** 取语料前 2,048 行建索引。这个服务在 CPU 上做 FAISS 查询，E5 编码仍使用 GPU。

~~~bash
head -n 2048 data/wiki18/wiki-18.jsonl >data/wiki18/smoke/corpus.jsonl
"$RETRIEVAL_PYTHON" search/index_builder.py
CORPUS_PATH=data/wiki18/smoke/corpus.jsonl INDEX_PATH=indexes/wiki18-smoke/e5_Flat.index "$RETRIEVAL_PYTHON" search/retrieval_http.py
~~~

在另一终端查询索引：

~~~bash
curl -sS http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' -d '{"queries":["Who was Evan Morris?"],"topk":3,"return_scores":true}'
~~~

**全量索引与单题 demo。** 停止小样本服务后建全量索引。下面的命令使用机器上列出的所有 GPU；也可以只把选定的 GPU UUID 传给 `--gpus`。建库完成后启动 GPU 检索服务，在另一终端运行 `python search/search_r1_demo.py`。

~~~bash
mapfile -t GPUS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
"$RETRIEVAL_PYTHON" search/index_builder_multi.py --gpus "${GPUS[@]}"
bash runs/grpo-20261120-144806/retrieval.sh
~~~

**训练与恢复。** 在上一步的前台检索服务退出并释放端口后，从仓库根目录执行以下命令。后台 supervisor 会启动本次检索服务、等待端口就绪，然后启动训练。中断后用同一命令从最近一次成功保存的 checkpoint 恢复；进度记录在该次运行目录的 `train.log` 和 `status.log`。

~~~bash
bash runs/grpo-20261120-144806/start-or-resume.sh
~~~

**全量测试。** 训练前的 baseline 使用 GPU 1 上的全量检索服务，以及 `bash scripts/start_vllm_multi.sh` 启动的五个独立模型服务。端口 8001–8005 就绪后执行 `python scripts/eval_baseline_multi.py`。训练后单独运行下面的入口，它从自己训练得到的第 720 步 checkpoint 合并权重，启动检索和四个推理服务，生成独立的结果目录。启动前释放 baseline 服务占用的端口和 GPU。

~~~bash
bash results/test-grpo-step720-20261122/run.sh
~~~

## 实验记录

| 项目 | 配置 |
|---|---|
| 模型与并行 | Qwen2.5-3B-Instruct 全参数 GRPO。4 张 RTX 3090（0、2、3、4）用于 FSDP 更新并承载 4 个 TP=1 的 vLLM 副本。GPU 1（Quadro RTX 8000）用于 E5 和 FAISS。 |
| 数据 | NQ 78,690 题 + HotpotQA 89,901 题参与训练。验证 1,024 题，种子 42。测试 51,713 题，7 个数据集。 |
| 更新 | 40 题 × 每题 4 条 = 160 条轨迹更新一次。PPO epoch 1，每卡 microbatch 2，旧策略 log prob 每卡 microbatch 8。 |
| 优化 | AdamW，学习率 1e-6，weight decay 0.01，warmup 实际 398 步。PPO clip 0.2，梯度裁剪 1.0；KL loss、KL 奖励和熵奖励均关闭。FSDP 参数和优化器状态使用 CPU offload。 |
| 生成 | 训练温度 1.0，最多检索 2 次，最多生成 3 轮，每轮 512 token。每轮读取最近 4,096 token，序列上限 4,608；每次 top-3 检索结果合计截到 500 token。vLLM 显存预算 0.90，每轮调度的 token 预算 8,192。 |
| 验证与测试 | 验证时每批 160 题，每题一条。训练前后的全量测试都用贪心解码，最多搜索 4 次、5 轮生成、8,192 上下文；训练前 5 卡 × 32 并发，第 720 步 4 卡 × 32 并发。 |

| 步数 | 0 | 180 | 360 | 540 | 720 |
|---|---:|---:|---:|---:|---:|
| 验证 EM | 21.68% | 32.81% | 39.75% | 40.53% | 41.11% |

计划 1,400 步，实际完成 873 步后，整机主机内存触及 Ray 的 95% 阈值。全量测试采用已保存的最佳验证 checkpoint（第 720 步）。按题干匹配，有 10 道测试题也出现在训练数据中；排除后，训练前后测试 EM 分别为 28.80% 和 42.17%。[训练前汇总](results/baseline-gpu-retrieval-20261117-211958/summary.json) · [第 720 步汇总](results/test-grpo-step720-20261122/summary.json)

## 来源与许可

多轮搜索实验参考 [Search-R1](https://github.com/PeterGriffinJin/Search-R1)，训练框架基于 [veRL 固定版本](https://github.com/verl-project/verl/tree/8fdc4d3f202f41461f4de9f42a637228e342668b)。适配目录中的 `ray_trainer.py` 与 `vllm_async_server.py` 在该版本基础上修改。上游 Apache-2.0 许可证与版权声明随附于 [third_party](third_party/)。数据和模型分别取自 [问答数据集](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train)、[Wiki 语料](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus)、[E5](https://huggingface.co/intfloat/e5-base-v2) 和 [Qwen](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)。
