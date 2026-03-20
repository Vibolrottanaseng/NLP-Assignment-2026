import re
import fitz
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Contextual Retrieval Chatbot",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1100px;
    }
    .stChatMessage {
        border-radius: 12px;
    }
    .source-box {
        padding: 12px;
        border-radius: 10px;
        background-color: rgba(127,127,127,0.08);
        border: 1px solid rgba(127,127,127,0.18);
        margin-top: 8px;
        white-space: pre-wrap;
    }
    .small-note {
        color: #666;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Utilities
# -----------------------------
def load_chapter_text(pdf_path):
    doc = fitz.open(pdf_path)
    start_page = 95   # printed page 96
    end_page = 119    # exclusive
    chapter_pages = []

    for page_num in range(start_page, end_page):
        chapter_pages.append(doc[page_num].get_text("text"))

    return "\n\n".join(chapter_pages)


def clean_text_v2(text: str) -> str:
    text = re.sub(r'\n\s*[•●]\s*\n', '\n', text)
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)
    text = re.sub(r'(?m)^\s*\d+\.\d+\s*$', '', text)
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)

    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        s = line.strip()
        if re.fullmatch(r'\.{2,}', s):
            continue
        if re.fullmatch(r'\d+\.', s):
            continue
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)
    text = re.sub(r'(?m)^[A-Z][A-Z\s\-]{5,}$', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' +\n', '\n', text)
    return text.strip()


@st.cache_resource
def load_local_model():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device


def enrich_chunk_local(chunk: str, document: str, title: str, tokenizer, model, device) -> str:
    doc_excerpt = document[:2500]

    prompt = f"""
You are writing a short contextual note for a chunk from a textbook chapter.

Chapter title: {title}

Document excerpt:
{doc_excerpt}

Chunk:
{chunk}

Write exactly one sentence that explains what this chunk is mainly about in relation to the chapter.
Start your sentence with:
This chunk from {title} discusses
""".strip()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False
    )

    context = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if not context.startswith(f"This chunk from {title} discusses"):
        context = f"This chunk from {title} discusses concepts related to word meaning and embeddings."

    if len(context) < 30:
        context = f"This chunk from {title} discusses concepts related to word meaning and embeddings."

    return f"{context}\n\n{chunk}"


@st.cache_resource
def load_contextual_retriever():
    pdf_path = "ed3book_jan26.pdf"
    raw_text = load_chapter_text(pdf_path)
    clean_chapter_text = clean_text_v2(raw_text)

    chapter_doc = Document(
        page_content=clean_chapter_text,
        metadata={"source": "ed3book_jan26.pdf", "chapter": 5, "title": "Embeddings"}
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    raw_chunks = text_splitter.split_documents([chapter_doc])

    naive_chunks = []
    for chunk in raw_chunks:
        text = chunk.page_content.strip()
        if len(text) < 120:
            continue
        if re.fullmatch(r"[A-Z][A-Z\s\-]+", text):
            continue
        naive_chunks.append(chunk)

    tokenizer, model, device = load_local_model()

    contextual_chunks = []
    for i, doc in enumerate(naive_chunks):
        enriched_text = enrich_chunk_local(
            chunk=doc.page_content,
            document=clean_chapter_text,
            title="Embeddings",
            tokenizer=tokenizer,
            model=model,
            device=device
        )

        contextual_chunks.append(
            Document(
                page_content=enriched_text,
                metadata={
                    **doc.metadata,
                    "chunk_id": i + 1
                }
            )
        )

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    contextual_vectorstore = FAISS.from_documents(contextual_chunks, embedding_model)

    contextual_retriever = contextual_vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return contextual_retriever


def clean_source_text(text: str) -> str:
    text = re.sub(r"^This chunk from Embeddings discusses.*?\.\s*", "", text)
    text = re.sub(r'\([^)]*\d{4}[^)]*\)', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def answer_from_context(question: str, retriever):
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return "I cannot find the answer in the chapter.", "No source chunk found."

    cleaned_docs = [clean_source_text(doc.page_content) for doc in docs]
    context = " ".join(cleaned_docs)

    sentences = re.split(r'(?<=[.!?])\s+', context)
    query_words = [w.lower() for w in re.findall(r'\w+', question) if len(w) > 2]

    scored = []
    for i, sent in enumerate(sentences):
        s = sent.strip()
        if not s:
            continue

        s_lower = s.lower()
        score = 0

        for w in query_words:
            if w in s_lower:
                score += 1

        if question.lower() in s_lower:
            score += 10

        if "is called" in s_lower or "is the study of" in s_lower:
            score += 3

        score += max(0, 3 - i * 0.1)
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    best_sentences = []
    seen = set()
    for score, sent in scored:
        if score <= 0:
            continue
        if sent not in seen:
            best_sentences.append(sent)
            seen.add(sent)
        if len(best_sentences) == 2:
            break

    if not best_sentences:
        answer = "I cannot find the answer in the chapter."
    else:
        answer = " ".join(best_sentences)

    top_doc = docs[0]
    source_chunk = clean_source_text(top_doc.page_content)

    return answer, source_chunk


# -----------------------------
# App UI
# -----------------------------
st.title("📘 Contextual Retrieval Chatbot")
st.caption("Ask questions about Chapter 5: Embeddings")


with st.spinner("Loading contextual retriever..."):
    contextual_retriever = load_contextual_retriever()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "source" in msg:
            st.markdown("**Source chunk used:**")
            st.markdown(f"<div class='source-box'>{msg['source']}</div>", unsafe_allow_html=True)

user_prompt = st.chat_input("Ask a question about Chapter 5...")

if user_prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt
    })

    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            answer, source_chunk = answer_from_context(user_prompt, contextual_retriever)

        st.write(answer)
        st.markdown("**Source chunk used:**")
        st.markdown(f"<div class='source-box'>{source_chunk}</div>", unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "source": source_chunk
    })