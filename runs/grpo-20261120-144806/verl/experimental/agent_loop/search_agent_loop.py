import asyncio
import re
from uuid import uuid4

import httpx

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput


SEARCH_URL = "http://127.0.0.1:8000/retrieve"
TOPK = 3
MAX_SEARCHES = 2
MAX_ROUNDS = MAX_SEARCHES + 1
MAX_OBS_TOKENS = 500
MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 512


async def search(query, client):
    """发送一个搜索词，返回可以追加到上下文的文档文本。"""
    for attempt in range(3):
        try:
            response = await client.post(
                SEARCH_URL,
                json={"queries": [query], "topk": TOPK, "return_scores": True},
            )
            response.raise_for_status()
            break
        except (httpx.NetworkError, httpx.RemoteProtocolError, httpx.TimeoutException):
            if attempt == 2:
                raise
            await asyncio.sleep(attempt + 1)

    documents = response.json()["result"][0]
    passages = []
    for rank, hit in enumerate(documents, start=1):
        title, _, body = hit["document"]["contents"].partition("\n")
        passages.append(f"Doc {rank}(Title: {title}) {body}")
    return "\n".join(passages)


class SearchAgentLoop(AgentLoopBase):
    async def run(self, messages, sampling_params):
        # 1. 初始提示词只套一次聊天模板。
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
        )

        # response_ids 保存模型输出和工具结果；mask 区分两者。
        response_ids = []
        response_mask = []

        rollout = self.config.actor_rollout_ref.rollout
        context_limit = rollout.max_model_len or (
            rollout.prompt_length + rollout.response_length
        )
        response_budget = min(
            rollout.response_length, context_limit - len(prompt_ids),
        )

        request_id = uuid4().hex
        search_count = 0
        model_turns = 0

        async with httpx.AsyncClient(trust_env=False, timeout=180.0) as client:
            for _ in range(MAX_ROUNDS):
                remaining = response_budget - len(response_ids)
                if remaining <= 0:
                    break

                # 2. 用 veRL 管理的模型生成，保留训练/验证传入的采样参数。
                params = dict(sampling_params)
                params.update(
                    max_tokens=min(MAX_NEW_TOKENS, remaining),
                    stop=["</search>", "</answer>"],
                    include_stop_str_in_output=True,
                )
                generated_ids = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=(prompt_ids + response_ids)[-MAX_INPUT_TOKENS:],
                    sampling_params=params,
                )
                model_turns += 1

                # 直接保留模型采样出的 token，不把生成文本重新编码。
                response_ids.extend(generated_ids)
                response_mask.extend([1] * len(generated_ids))

                # 3. 文本解码只用于判断模型要搜索还是作答。
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                match = re.search(r"<(search|answer)>(.*?)</\1>", text, re.DOTALL)
                if not match:
                    break

                action = match.group(1)
                content = match.group(2).strip()
                if not content or action == "answer":
                    break
                if search_count >= MAX_SEARCHES:
                    break
                if len(response_ids) >= response_budget:
                    break

                # 4. 查询检索服务，将返回文本编码成新的上下文 token。
                passages = await search(content, client)
                information = f"\n\n<information>{passages}</information>\n\n"
                # Match Search-R1: truncate the tokenized observation, including its tags.
                information_ids = self.tokenizer.encode(
                    information, add_special_tokens=False,
                )[:MAX_OBS_TOKENS]

                # 留下继续生成的空间；放不下时结束在模型输出处。
                if len(response_ids) + len(information_ids) >= response_budget:
                    break

                response_ids.extend(information_ids)
                response_mask.extend([0] * len(information_ids))
                search_count += 1

        # 5. 交给 veRL：原始提示词、完整轨迹、哪些 token 属于模型动作。
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            num_turns=1 + model_turns + search_count,
            metrics={},
        )
