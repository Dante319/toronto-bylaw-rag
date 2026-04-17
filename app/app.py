import streamlit as st
import logging
import sys
from pathlib import Path

# Ensure project root is on path when running via streamlit
sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.query_expand import expand_and_retrieve
from retrieval.generate import generate_answer
from retrieval.retrieve import get_embed_model
from retrieval.domain_detect import detect_domain
from config import DOMAIN_DETECTION_THRESHOLD, DOMAIN_DESCRIPTIONS

logging.basicConfig(level=logging.WARNING)


DOMAIN_LABELS = {
    "short_term_rental": "Short-term rentals (Ch. 547)",
    "noise": "Noise by-laws (Ch. 591)",
    None: "All domains",
}


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Toronto By-law Q&A",
    page_icon="🏙️",
    layout="centered",
)

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("🏙️ Toronto By-law Q&A")
st.caption(
    "Ask questions about Toronto's short-term rental regulations and noise by-laws. "
    "Answers are grounded in the Toronto Municipal Code with section citations."
)
st.divider()

# ── Example queries ────────────────────────────────────────────────────────────

EXAMPLES = [
    "Can I rent out my basement on Airbnb without a licence?",
    "What are the quiet hours for construction noise in Toronto?",
    "Can my landlord list my unit on Airbnb while I'm a tenant?",
    "What noise rules apply to outdoor music at a restaurant?",
    "What happens if I operate a short-term rental without registering?",
]

st.markdown("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for i, (col, example) in enumerate(zip(cols, EXAMPLES)):
    if col.button(f"Example {i+1}", key=f"ex_{i}", use_container_width=True):
        st.session_state["query_input"] = example

# ── Query input ────────────────────────────────────────────────────────────────

query = st.text_area(
    label="Your question",
    placeholder="e.g. Can I rent out my spare bedroom on Airbnb?",
    height=80,
    key="query_input",
    label_visibility="collapsed",
)

ask_col, clear_col = st.columns([4, 1])
ask = ask_col.button("Ask", type="primary", use_container_width=True)
if clear_col.button("Clear", use_container_width=True):
    st.session_state["query_input"] = ""
    st.rerun()

# ── Response ───────────────────────────────────────────────────────────────────

if ask and query.strip():
    model = get_embed_model()   # already cached — no reload
    domain, domain_scores = detect_domain(query, model)

    with st.spinner("Searching by-laws..."):
        chunks, expanded_query = expand_and_retrieve(
            query, top_k=5, domain=domain
        )
        result = generate_answer(query, chunks[:5])
    
    st.markdown("### Answer")
    st.markdown(result["answer"])

    if result["sources"]:
        st.markdown(
            "**Sections cited:** " +
            " · ".join(f"`§ {s}`" for s in sorted(result["sources"]))
        )

    st.divider()

    # Domain scores expander — replaces the hardcoded badge
    with st.expander("🗂️ Domain detection"):
        st.caption(
            "Domain was inferred by semantic similarity between your query "
            "and each domain's description. Queries below the confidence "
            f"threshold ({DOMAIN_DETECTION_THRESHOLD}) search all domains."
        )
        for d, score in sorted(domain_scores.items(), key=lambda x: -x[1]):
            label = DOMAIN_LABELS.get(d, d)
            st.progress(
                min(score, 1.0),
                text=f"{label}: `{score:.3f}`"
            )
        st.markdown(
            f"**Selected:** {DOMAIN_LABELS.get(domain, 'All domains')}"
        )

    st.divider()

    # ── Metadata row ───────────────────────────────────────────────────────────
    meta_cols = st.columns(3)

    meta_cols[0].metric(
        label="Domain searched",
        value=DOMAIN_LABELS[domain],
    )
    meta_cols[1].metric(
        label="Chunks retrieved",
        value=result["chunks_used"],
    )
    meta_cols[2].metric(
        label="Sections cited",
        value=len(result["sources"]),
    )

    # ── HyDE expanded query ────────────────────────────────────────────────────
    with st.expander("🔍 HyDE expanded query"):
        st.caption(
            "Before retrieving, your query was rewritten as a hypothetical "
            "by-law passage to improve retrieval over legal text."
        )
        st.info(expanded_query)

    # ── Retrieved source chunks ────────────────────────────────────────────────
    with st.expander(f"📄 Retrieved source chunks ({len(chunks)})"):
        st.caption(
            "These are the by-law passages the answer was generated from, "
            "ranked by relevance score."
        )
        for i, chunk in enumerate(chunks, 1):
            domain_badge = (
                "🏠 Short-term rental"
                if chunk.domain == "short_term_rental"
                else "🔊 Noise"
            )
            st.markdown(
                f"**{i}. § {chunk.section_id} — {chunk.section_title}**  "
                f"`{domain_badge}` · score `{chunk.score:.4f}`"
                + (f" · p. {chunk.page}" if chunk.page else "")
            )
            st.markdown(chunk.text.strip())
            if i < len(chunks):
                st.divider()

elif ask and not query.strip():
    st.warning("Please enter a question first.")

# ── Footer ─────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "⚠️ This tool is for informational purposes only and does not constitute "
    "legal advice. By-laws may have been amended — always verify with the "
    "[City of Toronto](https://toronto.ca) or a legal professional. · "
    "Built by Santosh Kolagati · [GitHub](https://github.com/Dante319/toronto-bylaw-rag)"
)