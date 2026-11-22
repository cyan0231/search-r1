import re
from pathlib import Path

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from search_client import search


MODEL_PATH = Path(__file__).resolve().parents[1] / "models/Qwen2.5-3B-Instruct"

QUESTION = (
    "Mike Barnett negotiated many contracts including which player "
    "that went on to become general manager of CSKA Moscow "
    "of the Kontinental Hockey League?"
)


@torch.no_grad()
def generate(context, tokenizer, model):
    inputs = tokenizer(
        context,
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")

    input_length = inputs["input_ids"].shape[1]

    if input_length >= 8192:
        raise RuntimeError("上下文已达到 8192 token，停止运行")

    output = model.generate(
        **inputs,
        max_new_tokens=min(512, 8192 - input_length),
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


def main():
    # 1. 加载模型
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).cuda().eval()

    # 2. 准备初始上下文
    instruction = (
        "Answer the question. "
        "Reason inside <think>...</think>. "
        "If you need information, write <search>your query</search> and stop. "
        "The search results will be provided inside <information>...</information>. "
        "Do not generate search results yourself. "
        "After receiving results, continue reasoning or search again. "
        "You may search at most 4 times. "
        "Put your final short answer inside <answer>...</answer>.\n"
        f"Question: {QUESTION}"
    )

    context = tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )

    with requests.Session() as session:
        session.trust_env = False
        search_count = 0

        for turn in range(1, 6):
            # 3. 模型生成
            text = generate(context, tokenizer, model)

            print(f"\n===== 第 {turn} 轮 =====")
            print(text)

            # 4. 提取搜索指令或答案
            match = re.search(
                r"<(search|answer)>(.*?)</\1>",
                text,
                flags=re.DOTALL,
            )

            if not match:
                print("没有生成完整的搜索或回答标签，停止。")
                break

            action = match.group(1)
            content = match.group(2).strip()

            if not content:
                print("搜索词或答案为空，停止。")
                break

            # 5. 追加模型输出
            context += text[:match.end()]

            if action == "answer":
                print(f"\n最终答案：{content}")
                break

            if search_count >= 4:
                print("已达到 4 次搜索上限，停止。")
                break

            # 6. 调用独立文件中的 search()
            print(f"\n搜索词：{content}")
            passages = search(content, session)
            search_count += 1

            print(f"\n检索结果：\n{passages}")

            # 7. 追加检索结果，进入下一轮
            context += (
                f"\n\n<information>{passages}</information>\n\n"
            )


if __name__ == "__main__":
    main()