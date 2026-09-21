"""Offline-first LangGraph RAG service. MOCK_LLM is enabled unless explicitly 0."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Literal, TypedDict
import chromadb
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

ROOT=Path(__file__).parent
KEYWORDS=("delivery","return","refund","membership","tracking","cancel","gift card","support hours")
PROMPT_TEMPLATE = """ROLE: You are Zepto's policy assistant.
CONTEXT: {context}
TASK: Answer the user's policy question: {query}
FORMAT: Return JSON with answer, sources, and confidence.
LENGTH: Use 2 concise sentences maximum.
CONSTRAINT: Do not answer using information not present in the provided context.
EXAMPLE: Context: 'Gift cards are valid for 1 year.' Question: 'How long are gift cards valid?' Answer: 'They are valid for 1 year.'
"""
class AskRequest(BaseModel): query: str = Field(min_length=1)
class AnswerResponse(BaseModel): answer: str; sources: list[str]; confidence: float = Field(ge=0, le=1)
class State(TypedDict, total=False): query: str; intent: Literal["policy_question","general_question"]; documents: list[str]; sources: list[str]; answer: str; confidence: float

def mock_mode() -> bool: return os.getenv("MOCK_LLM", "1") != "0"
def collection():
    client=chromadb.PersistentClient(path=str(ROOT/"chroma_db"))
    col=client.get_or_create_collection("zepto_policies", metadata={"hnsw:space":"cosine"})
    if col.count() == 0:
        model=SentenceTransformer("all-MiniLM-L6-v2")
        files=sorted((ROOT/"docs").glob("*.txt")); texts=[f.read_text(encoding="utf-8") for f in files]
        col.add(ids=[f.stem for f in files], documents=texts, embeddings=model.encode(texts).tolist(), metadatas=[{"source":f.stem} for f in files])
    return col
COLLECTION = collection()
EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

def classify_intent(state: State):
    # Optional real-LLM classification can replace this; the required deterministic heuristic is retained.
    intent="policy_question" if any(k in state["query"].lower() for k in KEYWORDS) else "general_question"
    return {"intent":intent}
def retrieve_and_answer(state: State):
    result=COLLECTION.query(query_embeddings=[EMBEDDER.encode(state["query"]).tolist()], n_results=3, include=["documents"])
    docs=result["documents"][0]; ids=result["ids"][0]
    if mock_mode(): return {"documents":docs,"sources":ids,"answer":f"Based on the retrieved context: {docs[0][:200]}","confidence":1.0}
    # Optional extension: wire a provider call here and validate/retry its JSON up to two times.
    for attempt in range(3):
        try:
            import json
            from openai import OpenAI
            raw=OpenAI(base_url="https://api.groq.com/openai/v1",api_key=os.environ["GROQ_API_KEY"]).chat.completions.create(model="llama-3.1-8b-instant",messages=[{"role":"user","content":PROMPT_TEMPLATE.format(context="\n".join(docs),query=state["query"])}],response_format={"type":"json_object"}).choices[0].message.content
            parsed=AnswerResponse.model_validate_json(raw); return {"documents":docs,"sources":parsed.sources or ids,"answer":parsed.answer,"confidence":parsed.confidence}
        except Exception as exc: last_error=str(exc)
    return {"documents":docs,"sources":ids,"answer":f"ERROR: real LLM response could not validate after retries: {last_error}","confidence":0.0}
def direct_answer(state: State):
    if mock_mode(): return {"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
    return {"answer":"Real LLM direct answers are configured through the optional provider path.","sources":[],"confidence":0.5}
def route(state: State): return "retrieve_and_answer" if state["intent"]=="policy_question" else "direct_answer"

graph=StateGraph(State); graph.add_node("classify_intent",classify_intent); graph.add_node("retrieve_and_answer",retrieve_and_answer); graph.add_node("direct_answer",direct_answer); graph.set_entry_point("classify_intent"); graph.add_conditional_edges("classify_intent",route); graph.add_edge("retrieve_and_answer",END); graph.add_edge("direct_answer",END); app_graph=graph.compile()
app=FastAPI(title="Zepto Policy Assistant")
@app.post("/ask",response_model=AnswerResponse)
def ask(request: AskRequest):
    state=app_graph.invoke({"query":request.query})
    return AnswerResponse(answer=state["answer"],sources=state["sources"],confidence=state["confidence"])
