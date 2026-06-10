# app.py
# LexiQuery — AI Contract Review Chatbot
# Streamlit frontend: PDF upload + chat + clause summary + risk flags

import streamlit as st
import tempfile
import os
from pipeline.embedder import embed_pdf, get_embedding_model
from pipeline.chain import answer_question
from features.clause_extractor import extract_clauses
from features.risk_flagger import flag_risks

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexiQuery — AI Contract Review",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ LexiQuery")
st.caption("Upload a contract. Ask it anything. Get grounded answers with clause citations.")

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "clauses" not in st.session_state:
    st.session_state.clauses = []
if "risks" not in st.session_state:
    st.session_state.risks = []
if "model" not in st.session_state:
    with st.spinner("Loading legal-BERT model (first run only, ~400MB)..."):
        st.session_state.model = get_embedding_model()

# ── Layout: two columns ───────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.6], gap="large")

# ══════════════════════════════════════════════
# LEFT COLUMN — Upload + Contract Summary
# ══════════════════════════════════════════════
with left_col:
    st.subheader("📄 Upload Contract")
    uploaded_file = st.file_uploader(
        "Drop a PDF contract here",
        type=["pdf"],
        help="Supports NDAs, employment agreements, leases, vendor contracts, etc."
    )

    if uploaded_file:
        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Process the PDF (embed once, cache collection name in session)
        if st.session_state.collection_name is None:
            with st.spinner("Reading and embedding contract..."):
                st.session_state.collection_name = embed_pdf(tmp_path, st.session_state.model)

            with st.spinner("Extracting clause summary..."):
                st.session_state.clauses = extract_clauses(tmp_path)

            with st.spinner("Scanning for risk patterns..."):
                st.session_state.risks = flag_risks(tmp_path)

            st.success(f"✅ Contract ready — {len(st.session_state.clauses)} key clauses detected")

        # ── Clause summary ────────────────────────────────
        if st.session_state.clauses:
            st.subheader("🗂️ Key Clauses Detected")
            for clause in st.session_state.clauses:
                with st.expander(clause["type"]):
                    st.write(clause["summary"])

        # ── Risk flags ────────────────────────────────────
        if st.session_state.risks:
            st.subheader("🚨 Potential Risks")
            for risk in st.session_state.risks:
                with st.expander(risk["label"]):
                    st.write(risk["description"])
                    for snippet in risk["snippets"]:
                        st.code(snippet, language=None)
        elif st.session_state.clauses:
            st.success("✅ No common risk patterns detected")

    else:
        st.info("Upload a PDF contract to get started.")
        # Reset on new session
        st.session_state.collection_name = None
        st.session_state.chat_history = []
        st.session_state.clauses = []
        st.session_state.risks = []

# ══════════════════════════════════════════════
# RIGHT COLUMN — Chat Interface
# ══════════════════════════════════════════════
with right_col:
    st.subheader("💬 Ask the Contract")

    # Suggested starter questions
    if not st.session_state.chat_history and st.session_state.collection_name:
        st.write("**Try asking:**")
        starter_questions = [
            "What is the termination notice period?",
            "Is there a non-compete clause? What does it say?",
            "Who owns intellectual property created during this agreement?",
            "What happens if either party breaches the contract?",
            "What is the governing law?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(starter_questions):
            if cols[i % 2].button(q, key=f"starter_{i}", use_container_width=True):
                st.session_state.pending_question = q

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for turn in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(turn["question"])
            with st.chat_message("assistant"):
                st.write(turn["answer"])
                if turn.get("sources"):
                    with st.expander(f"📎 {len(turn['sources'])} source clause(s) used"):
                        for src in turn["sources"]:
                            st.markdown(f"**[{src['chunk_id']}]** (relevance: {src['score']})")
                            st.code(src["text"][:400] + "..." if len(src["text"]) > 400 else src["text"])
                if not turn.get("found_in_doc"):
                    st.warning("⚠️ Answer not found in document — consult a lawyer.")

    # Chat input
    question = st.chat_input(
        "Ask anything about this contract...",
        disabled=st.session_state.collection_name is None
    )

    # Handle starter question button clicks
    if hasattr(st.session_state, "pending_question"):
        question = st.session_state.pending_question
        del st.session_state.pending_question

    if question and st.session_state.collection_name:
        with st.spinner("Searching contract..."):
            result = answer_question(
                question=question,
                collection_name=st.session_state.collection_name,
                model=st.session_state.model
            )

        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
            "found_in_doc": result["found_in_doc"]
        })
        st.rerun()

    elif question and not st.session_state.collection_name:
        st.warning("Please upload a contract first.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ LexiQuery is an educational portfolio project and is NOT a substitute for professional legal advice. "
    "Always consult a qualified lawyer for actual legal matters."
)