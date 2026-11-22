SEARCH_URL = "http://127.0.0.1:8000/retrieve"
## 用于请求http服务 并且返回内容

def search(query, session):
    response = session.post(
        SEARCH_URL,
        json={
            "queries": [query],
            "topk": 3,
            "return_scores": True,
        },
        timeout=180,
    )
    response.raise_for_status()

    documents = response.json()["result"][0]

    passages = []
    for rank, hit in enumerate(documents, start=1):
        title, _, body = hit["document"]["contents"].partition("\n")
        passages.append(f"Doc {rank}(Title: {title}) {body}")

    return "\n".join(passages)