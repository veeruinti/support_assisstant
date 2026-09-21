# Support Assistant

Run from the repository root: `uvicorn support_assistant.main:app --port 7860`. Mock mode is the default (`MOCK_LLM` unset or `1`), so no LLM key or network LLM call is made. Example calls:

```powershell
Invoke-RestMethod -Method Post http://localhost:7860/ask -ContentType application/json -Body '{"query":"What is the delivery fee?"}'
# {"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01","doc_05","doc_02"],"confidence":1.0}
Invoke-RestMethod -Method Post http://localhost:7860/ask -ContentType application/json -Body '{"query":"Tell me a joke"}'
# {"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

Architecture: ingestion is `collection()` reading the eight files in `docs/`; it embeds one chunk per document with `all-MiniLM-L6-v2` and persists vectors to the `zepto_policies` ChromaDB collection. `classify_intent` routes keyword policy questions through the conditional LangGraph edge. `retrieve_and_answer` embeds the query and retrieves top-three cosine-similar chunks; it generates the grounded answer. `direct_answer` handles unrelated queries. Generation branches on `MOCK_LLM`: mock mode emits deterministic answers, while `MOCK_LLM=0` uses the optional Groq-compatible branch with schema-validation retries. The full role/context/task/format/length prompt is in `PROMPT_TEMPLATE`.

Build/run container: `docker build -f support_assistant/Dockerfile -t zepto-assistant .` then `docker run -p 7860:7860 zepto-assistant`. The first container start downloads the open-source MiniLM embedding model if it is not already cached in the container.
