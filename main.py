import os

import streamlit as st

from llm.infer import LLM
from assistant.chatbot import Chatbot
from assistant.chat_store import ChatStore
from model_list import MODELS
from assistant.rag import RAG

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")

if "chat_store" not in st.session_state:
    st.session_state.chat_store = ChatStore()

store: ChatStore = st.session_state.chat_store


def format_sources(chunks: list[dict]) -> list[dict]:
    sources = []
    for c in chunks:
        meta = c.get("metadata", {})
        sources.append(
            {
                "filename": meta.get("filename", "unknown"),
                "section": meta.get("section_title") or meta.get("title", ""),
                "score": c.get("rerank_score", 0),
                "snippet": (c.get("text") or "")[:300],
            }
        )
    return sources


def render_sources_from_stored(sources: list[dict]):
    if not sources:
        return
    with st.expander("Sources", expanded=False):
        for s in sources:
            st.markdown(f"**{s['filename']}** · {s['section']} · score `{s['score']:.2f}`")
            snippet = s.get("snippet", "")
            st.caption(snippet + ("…" if len(snippet) >= 300 else ""))


def render_sources_from_chunks(chunks: list[dict]):
    render_sources_from_stored(format_sources(chunks))


def load_conversation(conversation_id: int):
    conv = store.get_conversation(conversation_id)
    if not conv:
        return
    st.session_state.conversation_id = conversation_id
    st.session_state.messages = store.get_messages(conversation_id)
    st.session_state.awaiting_clarification = bool(conv.get("awaiting_clarification"))
    st.session_state.original_query = conv.get("original_query") or ""
    st.session_state.loaded_conv_id = conversation_id


def ensure_conversation(backend: str, model: str) -> int:
    if "conversation_id" not in st.session_state:
        conversations = store.list_conversations()
        if conversations:
            load_conversation(conversations[0]["id"])
        else:
            conv_id = store.create_conversation(backend=backend, model=model)
            load_conversation(conv_id)
    elif "loaded_conv_id" not in st.session_state or st.session_state.loaded_conv_id != st.session_state.conversation_id:
        load_conversation(st.session_state.conversation_id)
    return st.session_state.conversation_id


with st.sidebar:
    st.header("Conversations")

    if st.button("＋ New chat", use_container_width=True):
        conv_id = store.create_conversation(
            backend=st.session_state.get("backend", "openai"),
            model=st.session_state.get("model", ""),
        )
        load_conversation(conv_id)
        st.rerun()

    conversations = store.list_conversations()
    active_id = st.session_state.get("conversation_id")

    for conv in conversations:
        label = conv["title"]
        updated = conv.get("updated_at", "")[:16].replace("T", " ")
        is_active = conv["id"] == active_id
        btn_label = f"{'● ' if is_active else ''}{label}"
        if st.button(btn_label, key=f"conv_{conv['id']}", use_container_width=True):
            load_conversation(conv["id"])
            st.rerun()
        st.caption(updated)

    if active_id and st.button("Delete chat", use_container_width=True):
        store.delete_conversation(active_id)
        remaining = store.list_conversations()
        if remaining:
            load_conversation(remaining[0]["id"])
        else:
            conv_id = store.create_conversation()
            load_conversation(conv_id)
        st.rerun()

    st.divider()
    st.header("Model")
    backend = st.selectbox("Backend", list(MODELS.keys()))
    model = st.selectbox("Model", MODELS[backend])

    st.divider()
    st.subheader("Documents")
    uploads = st.file_uploader(
        "Add files",
        type=["pdf", "docx", "txt", "md", "csv", "xlsx", "eml"],
        accept_multiple_files=True,
    )
    if uploads and st.button("Ingest"):
        if "rag" not in st.session_state:
            st.session_state.rag = RAG()
        progress = st.progress(0.0)
        status = st.empty()
        for i, uf in enumerate(uploads, 1):
            dest = os.path.join("data", uf.name)
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            result = next(st.session_state.rag.ingest([dest]))
            status.write(f"{uf.name}: **{result['status']}**")
            progress.progress(i / len(uploads))
        st.success("Done.")

if (
    "llm" not in st.session_state
    or st.session_state.get("backend") != backend
    or st.session_state.get("model") != model
):
    st.session_state.llm = LLM(backend, model=model)
    st.session_state.backend = backend
    st.session_state.model = model

llm = st.session_state.llm

if "awaiting_clarification" not in st.session_state:
    st.session_state.awaiting_clarification = False
if "original_query" not in st.session_state:
    st.session_state.original_query = ""

ensure_conversation(backend, model)
conversation_id = st.session_state.conversation_id

st.title("RAG Chatbot")
st.caption(f"Using **{backend}** / `{model}` · conversation #{conversation_id}")

if "rag" not in st.session_state:
    st.session_state.rag = RAG()

if (
    "chatbot" not in st.session_state
    or st.session_state.get("backend") != backend
    or st.session_state.get("model") != model
):
    st.session_state.chatbot = Chatbot(llm, rag=st.session_state.rag)

chatbot = st.session_state.chatbot

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources_from_stored(msg["sources"])

if prompt := st.chat_input("Ask anything..."):
    force_rag = False
    if st.session_state.awaiting_clarification:
        combined = f"{st.session_state.original_query} — specifically: {prompt}"
        st.session_state.awaiting_clarification = False
        st.session_state.original_query = ""
        query = combined
        force_rag = True
        store.update_conversation(
            conversation_id,
            awaiting_clarification=False,
            original_query="",
        )
    else:
        query = prompt

    history = list(st.session_state.messages)
    store.add_message(conversation_id, "user", prompt)

    if store.message_count(conversation_id) == 1:
        title = prompt.strip()[:50] + ("…" if len(prompt.strip()) > 50 else "")
        store.update_conversation(conversation_id, title=title or "New chat")

    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        result = chatbot.respond(query, history=history, force_rag=force_rag)

        if result.kind == "clarify":
            st.markdown(result.text)
            st.session_state.awaiting_clarification = True
            st.session_state.original_query = query
            store.update_conversation(
                conversation_id,
                awaiting_clarification=True,
                original_query=query,
            )
            assistant_msg = {
                "role": "assistant",
                "content": result.text,
                "used_rag": result.used_rag,
                "sources": [],
            }
            store.add_message(
                conversation_id,
                "assistant",
                result.text,
                used_rag=result.used_rag,
            )
            st.session_state.messages.append(assistant_msg)

        else:
            placeholder = st.empty()
            response = ""
            for chunk in chatbot.stream_response(query, result, history=history):
                response += chunk
                placeholder.markdown(response + "▌")
            placeholder.markdown(response)

            sources = format_sources(chatbot.last_chunks) if result.used_rag and result.kind == "answer" else []
            if sources:
                render_sources_from_stored(sources)

            assistant_msg = {
                "role": "assistant",
                "content": response,
                "used_rag": result.used_rag,
                "sources": sources,
            }
            store.add_message(
                conversation_id,
                "assistant",
                response,
                used_rag=result.used_rag,
                sources=sources,
            )
            st.session_state.messages.append(assistant_msg)
