"""
Dream Journal Analyzer — Streamlit Dashboard
=============================================
✨ Dreamy Edition ✨
Run with: streamlit run app.py
"""

import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from journal import DreamJournal
from nlp_pipeline import process_dream, USE_TRANSFORMERS
from analyzer import (
    symbol_frequency,
    symbol_co_occurrence,
    emotion_timeline,
    dominant_emotion_per_dream,
    temporal_patterns,
    find_similar_dreams,
    find_recurring_dreams,
    generate_insights,
)
from visualization import (
    emotion_timeline_chart,
    emotion_radar_chart,
    symbol_bar_chart,
    co_occurrence_network,
    dream_word_cloud,
    dominant_emotion_pie,
)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dreamscape — Your Dream Journal",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dreamy CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400&family=Quicksand:wght@300;400;500;600&display=swap');

/* ── Global ─────────────────────────────────────────── */
.stApp {
    background: radial-gradient(ellipse at 20% 50%, #0f0c29 0%, #090818 40%, #050510 100%);
    font-family: 'Quicksand', sans-serif;
}

/* Starfield background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    background-image:
        radial-gradient(2px 2px at 20px 30px, rgba(255,255,255,0.15), transparent),
        radial-gradient(2px 2px at 40px 70px, rgba(200,180,255,0.12), transparent),
        radial-gradient(1px 1px at 90px 40px, rgba(255,255,255,0.18), transparent),
        radial-gradient(1px 1px at 130px 80px, rgba(180,200,255,0.1), transparent),
        radial-gradient(2px 2px at 160px 30px, rgba(255,255,255,0.12), transparent),
        radial-gradient(1px 1px at 200px 60px, rgba(220,200,255,0.15), transparent);
    background-size: 200px 100px;
    pointer-events: none;
    z-index: 0;
    animation: twinkle 8s ease-in-out infinite alternate;
}

@keyframes twinkle {
    0% { opacity: 0.4; }
    50% { opacity: 0.8; }
    100% { opacity: 0.5; }
}

/* ── Sidebar ────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0b20 0%, #13102a 50%, #0a0818 100%) !important;
    border-right: 1px solid rgba(184, 169, 212, 0.08);
}

section[data-testid="stSidebar"] .stMarkdown h1 {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 300;
    letter-spacing: 3px;
    font-size: 1.6rem;
    color: #d4c8ef;
    text-align: center;
}

/* ── Headers ────────────────────────────────────────── */
h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 300 !important;
    letter-spacing: 2px;
    color: #e0d8f0 !important;
}

h1 {
    font-size: 2.4rem !important;
    background: linear-gradient(135deg, #e0d8f0 0%, #b8a9d4 50%, #8b7bb5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

h2 { font-size: 1.8rem !important; }
h3 { font-size: 1.4rem !important; color: #c4b8e0 !important; }

/* ── Body text ──────────────────────────────────────── */
p, li, span, label, .stMarkdown {
    font-family: 'Quicksand', sans-serif;
    color: #c8c0d8;
}

/* ── Dream card ─────────────────────────────────────── */
.dream-card {
    background: linear-gradient(145deg, rgba(20, 15, 45, 0.8), rgba(15, 12, 35, 0.9));
    border: 1px solid rgba(184, 169, 212, 0.12);
    border-radius: 16px;
    padding: 24px 28px;
    margin: 14px 0;
    backdrop-filter: blur(10px);
    box-shadow:
        0 4px 30px rgba(100, 70, 180, 0.06),
        inset 0 1px 0 rgba(255, 255, 255, 0.03);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.dream-card::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 200%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(184, 169, 212, 0.03), transparent);
    transition: left 0.6s ease;
}

.dream-card:hover {
    border-color: rgba(184, 169, 212, 0.25);
    box-shadow: 0 8px 40px rgba(100, 70, 180, 0.12);
    transform: translateY(-2px);
}

.dream-card:hover::before {
    left: 100%;
}

.dream-card small {
    color: #8b7bb5;
    font-family: 'Quicksand', sans-serif;
    font-size: 0.82rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Insight box ────────────────────────────────────── */
.insight-box {
    background: linear-gradient(135deg, rgba(25, 18, 50, 0.7), rgba(35, 20, 60, 0.6));
    border-left: 3px solid rgba(184, 169, 212, 0.4);
    padding: 18px 22px;
    margin: 12px 0;
    border-radius: 0 12px 12px 0;
    font-family: 'Quicksand', sans-serif;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #d4c8ef;
    backdrop-filter: blur(5px);
}

/* ── Hero section ───────────────────────────────────── */
.hero {
    text-align: center;
    padding: 30px 20px;
    margin-bottom: 20px;
}

.hero h1 {
    font-size: 3rem !important;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-weight: 300;
    color: #8b7bb5;
    font-size: 1.1rem;
    letter-spacing: 1.5px;
}

/* ── Metric cards ───────────────────────────────────── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(20, 15, 45, 0.6), rgba(12, 10, 28, 0.8));
    border: 1px solid rgba(184, 169, 212, 0.1);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}

[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 400;
    color: #d4c8ef !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'Quicksand', sans-serif;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #8b7bb5 !important;
}

/* ── Tabs ───────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid rgba(184, 169, 212, 0.1);
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Quicksand', sans-serif;
    font-weight: 500;
    letter-spacing: 1px;
    border-radius: 10px 10px 0 0;
    color: #8b7bb5;
    padding: 10px 20px;
}

.stTabs [aria-selected="true"] {
    background: rgba(184, 169, 212, 0.08) !important;
    color: #d4c8ef !important;
}

/* ── Text area ──────────────────────────────────────── */
.stTextArea textarea {
    background: rgba(15, 12, 30, 0.8) !important;
    border: 1px solid rgba(184, 169, 212, 0.15) !important;
    border-radius: 12px !important;
    color: #d4c8ef !important;
    font-family: 'Quicksand', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.8 !important;
    padding: 16px !important;
}

.stTextArea textarea:focus {
    border-color: rgba(184, 169, 212, 0.35) !important;
    box-shadow: 0 0 20px rgba(100, 70, 180, 0.1) !important;
}

.stTextArea textarea::placeholder {
    color: rgba(184, 169, 212, 0.3) !important;
    font-style: italic;
}

/* ── Text input ─────────────────────────────────────── */
.stTextInput input {
    background: rgba(15, 12, 30, 0.8) !important;
    border: 1px solid rgba(184, 169, 212, 0.15) !important;
    border-radius: 10px !important;
    color: #d4c8ef !important;
    font-family: 'Quicksand', sans-serif !important;
}

/* ── Buttons ────────────────────────────────────────── */
.stButton > button {
    font-family: 'Quicksand', sans-serif;
    font-weight: 500;
    letter-spacing: 1.5px;
    border-radius: 12px;
    transition: all 0.3s ease;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2d1b4e 0%, #1a1040 100%) !important;
    border: 1px solid rgba(184, 169, 212, 0.25) !important;
    color: #d4c8ef !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #3d2b5e 0%, #2a2050 100%) !important;
    border-color: rgba(184, 169, 212, 0.4) !important;
    box-shadow: 0 4px 20px rgba(100, 70, 180, 0.2);
}

/* ── Progress bars ──────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #2d1b4e, #6a4fa0, #b8a9d4) !important;
    border-radius: 8px;
}

/* ── Dividers ───────────────────────────────────────── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(184, 169, 212, 0.2), transparent);
    margin: 24px 0;
}

/* ── Scrollbar ──────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(184, 169, 212, 0.2);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(184, 169, 212, 0.35);
}

/* ── Symbol pills ───────────────────────────────────── */
.symbol-pill {
    display: inline-block;
    background: rgba(184, 169, 212, 0.08);
    border: 1px solid rgba(184, 169, 212, 0.15);
    border-radius: 20px;
    padding: 4px 14px;
    margin: 3px 4px;
    font-size: 0.85rem;
    color: #c4b8e0;
    font-family: 'Quicksand', sans-serif;
    transition: all 0.3s ease;
}

.symbol-pill:hover {
    background: rgba(184, 169, 212, 0.15);
    border-color: rgba(184, 169, 212, 0.3);
}

/* ── Emotion badge ──────────────────────────────────── */
.emotion-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(20, 15, 40, 0.6);
    border: 1px solid rgba(184, 169, 212, 0.1);
    border-radius: 10px;
    padding: 8px 16px;
    margin: 4px;
    font-size: 0.88rem;
    color: #c4b8e0;
}

/* ── Floating orbs (decorative) ─────────────────────── */
.orb {
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    filter: blur(60px);
}

.orb-1 {
    width: 300px; height: 300px;
    background: rgba(100, 60, 180, 0.04);
    top: 10%; right: 5%;
    animation: float1 20s ease-in-out infinite;
}

.orb-2 {
    width: 200px; height: 200px;
    background: rgba(60, 80, 180, 0.03);
    bottom: 20%; left: 10%;
    animation: float2 25s ease-in-out infinite;
}

.orb-3 {
    width: 250px; height: 250px;
    background: rgba(140, 80, 160, 0.03);
    top: 60%; right: 30%;
    animation: float3 18s ease-in-out infinite;
}

@keyframes float1 {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(-30px, 20px); }
}

@keyframes float2 {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(20px, -30px); }
}

@keyframes float3 {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(-20px, -20px); }
}
</style>

<!-- Floating orbs -->
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>
""", unsafe_allow_html=True)


# ── Initialize journal ─────────────────────────────────────────────────────
@st.cache_resource
def get_journal():
    return DreamJournal()

journal = get_journal()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 20px 0 10px 0;">
            <div style="font-size: 2.5rem; margin-bottom: 8px;">🌙</div>
            <h1 style="margin: 0; font-size: 1.4rem;">DREAMSCAPE</h1>
            <p style="color: #6b5b8a; font-family: 'Cormorant Garamond', serif; font-style: italic; font-size: 0.9rem; margin-top: 4px;">
                explore the landscape of your mind
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["✦ Log a Dream", "✦ Dashboard", "✦ Explore Dreams", "✦ Insights"],
        index=1,
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(f"""
        <div style="text-align: center; padding: 10px 0;">
            <div style="font-family: 'Cormorant Garamond', serif; font-size: 2rem; color: #d4c8ef;">
                {len(journal.entries)}
            </div>
            <div style="font-family: 'Quicksand', sans-serif; font-size: 0.7rem; color: #6b5b8a; letter-spacing: 3px; text-transform: uppercase;">
                dreams captured
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if USE_TRANSFORMERS:
        st.caption("🧠 AI models active")
    else:
        st.caption("⚡ Lightweight mode")

    if st.button("✦ Load Sample Dreams", use_container_width=True):
        samples = journal.get_sample_dreams()
        existing_texts = {e.text.strip() for e in journal.entries}
        count = 0
        for sample in samples:
            if sample["text"].strip() not in existing_texts:
                entry = journal.add_dream(
                    text=sample["text"],
                    dream_date=sample["date"],
                    tags=sample.get("tags", []),
                )
                result = process_dream(entry.text)
                entry.symbols = result["symbol_names"]
                entry.entities = result["entities"]
                entry.emotions = result["emotions"]
                entry.embedding = result["embedding"]
                journal.update_entry(entry)
                count += 1
        if count > 0:
            st.success(f"✨ {count} dreams materialized")
            st.rerun()
        else:
            st.info("Sample dreams already present")

    if len(journal.entries) > 0:
        st.markdown("")
        if st.button("🗑️ Clear All Dreams", use_container_width=True):
            st.session_state["confirm_clear"] = True

        if st.session_state.get("confirm_clear", False):
            st.warning("This will delete all dreams. Are you sure?")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Yes, clear", use_container_width=True):
                    journal.entries = []
                    journal.save()
                    st.session_state["confirm_clear"] = False
                    st.rerun()
            with col_no:
                if st.button("Cancel", use_container_width=True):
                    st.session_state["confirm_clear"] = False
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# ✦ LOG A DREAM
# ══════════════════════════════════════════════════════════════════════════
if page == "✦ Log a Dream":
    st.markdown("""
        <div class="hero">
            <h1>Record Your Dream</h1>
            <div class="hero-subtitle">capture it before it fades...</div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col1:
        dream_text = st.text_area(
            "Describe your dream",
            height=220,
            placeholder="Close your eyes and recall... what did you see?",
            label_visibility="collapsed",
        )

    with col2:
        dream_date = st.date_input("When did you dream this?")
        tags_input = st.text_input(
            "Tags",
            placeholder="lucid, recurring...",
        )
        st.markdown("<br>", unsafe_allow_html=True)

    if st.button("✦  Save & Analyze  ✦", type="primary", use_container_width=True):
        if dream_text.strip():
            tags = [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else []

            with st.spinner("✨ Interpreting the dream..."):
                entry = journal.add_dream(
                    text=dream_text,
                    dream_date=dream_date.isoformat(),
                    tags=tags,
                )
                result = process_dream(dream_text)
                entry.symbols = result["symbol_names"]
                entry.entities = result["entities"]
                entry.emotions = result["emotions"]
                entry.embedding = result["embedding"]
                journal.update_entry(entry)

            st.markdown("---")
            st.markdown("### ✨ Dream Analysis")

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### Symbols Discovered")
                if result["symbols"]:
                    pills_html = ""
                    for sym in result["symbols"]:
                        icon = {"dream_dictionary": "🔮", "spacy_ner": "👤", "spacy_pos": "◈"}.get(sym["source"], "◈")
                        pills_html += f'<span class="symbol-pill">{icon} {sym["text"]}</span>'
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.markdown("*No specific symbols detected*")

            with col_b:
                st.markdown("#### Emotional Tone")
                if result["emotions"]:
                    sorted_emos = sorted(result["emotions"].items(), key=lambda x: x[1], reverse=True)
                    for emo, score in sorted_emos[:5]:
                        emoji_map = {
                            "joy": "☀️", "fear": "👁️", "anxiety": "🌊", "sadness": "🌧️",
                            "anger": "🔥", "wonder": "✨", "confusion": "🌀", "peace": "🍃",
                            "love": "💜", "power": "⚡",
                        }
                        icon = emoji_map.get(emo, "◈")
                        st.markdown(f"{icon} **{emo.capitalize()}** — {score:.0%}")
                        st.progress(score)

            if entry.embedding:
                similar = find_similar_dreams(entry, journal.entries, top_n=3)
                if similar:
                    st.markdown("---")
                    st.markdown("### 🔗 Echoes of Past Dreams")
                    for s in similar:
                        st.markdown(
                            f'<div class="dream-card">'
                            f'<small>{s["date"]} · {s["similarity"]:.0%} resonance</small><br><br>'
                            f'{s["preview"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
        else:
            st.warning("Describe your dream first...")


# ══════════════════════════════════════════════════════════════════════════
# ✦ DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
elif page == "✦ Dashboard":
    st.markdown("""
        <div class="hero">
            <h1>Dreamscape</h1>
            <div class="hero-subtitle">patterns hidden in the night</div>
        </div>
    """, unsafe_allow_html=True)

    entries = journal.get_all()

    if not entries:
        st.markdown("""
            <div style="text-align: center; padding: 60px 20px; color: #6b5b8a;">
                <div style="font-size: 3rem; margin-bottom: 20px;">🌑</div>
                <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.3rem; font-style: italic;">
                    No dreams recorded yet.<br>
                    <span style="font-size: 0.95rem;">Log your first dream or load sample dreams to begin.</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Stats row
        freq = symbol_frequency(entries)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Dreams", len(entries))
        with col2:
            st.metric("Symbols", len(freq))
        with col3:
            most_common_sym = freq.most_common(1)[0][0] if freq else "—"
            st.metric("Top Symbol", most_common_sym)
        with col4:
            from collections import Counter
            emo_counts = Counter()
            for e in entries:
                if e.emotions:
                    top = max(e.emotions.items(), key=lambda x: x[1])[0]
                    emo_counts[top] += 1
            top_emo = emo_counts.most_common(1)[0][0].capitalize() if emo_counts else "—"
            st.metric("Top Emotion", top_emo)

        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs([
            "🌊  Emotions",
            "🔮  Symbols",
            "🕸️  Connections",
            "☁️  Word Cloud",
        ])

        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                fig = emotion_timeline_chart(entries)
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Quicksand", color="#c4b8e0"),
                    title_font=dict(family="Cormorant Garamond", size=20),
                )
                st.plotly_chart(fig, use_container_width=True, key="chart_emotion_timeline")
            with col_b:
                fig = emotion_radar_chart(entries)
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Quicksand", color="#c4b8e0"),
                    title_font=dict(family="Cormorant Garamond", size=20),
                )
                st.plotly_chart(fig, use_container_width=True, key="chart_emotion_radar")

            fig = dominant_emotion_pie(entries)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Quicksand", color="#c4b8e0"),
                title_font=dict(family="Cormorant Garamond", size=20),
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_emotion_pie")

        with tab2:
            fig = symbol_bar_chart(entries)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Quicksand", color="#c4b8e0"),
                title_font=dict(family="Cormorant Garamond", size=20),
            )
            fig.update_traces(marker_color="rgba(184, 169, 212, 0.6)")
            st.plotly_chart(fig, use_container_width=True, key="chart_symbol_bar")

        with tab3:
            fig = co_occurrence_network(entries, min_count=1)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Quicksand", color="#c4b8e0"),
                title_font=dict(family="Cormorant Garamond", size=20),
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_co_occurrence")

            co_occ = symbol_co_occurrence(entries, min_count=1)
            if co_occ:
                st.markdown("#### Linked Symbols")
                for pair in co_occ[:10]:
                    st.markdown(
                        f'<span class="symbol-pill">🔗 {pair["symbol_a"]} ↔ {pair["symbol_b"]} ({pair["count"]})</span>',
                        unsafe_allow_html=True,
                    )

        with tab4:
            wc_fig = dream_word_cloud(entries)
            if wc_fig:
                st.pyplot(wc_fig)
            else:
                st.markdown("*Not enough dreams for a word cloud yet...*")


# ══════════════════════════════════════════════════════════════════════════
# ✦ EXPLORE DREAMS
# ══════════════════════════════════════════════════════════════════════════
elif page == "✦ Explore Dreams":
    st.markdown("""
        <div class="hero">
            <h1>Explore Your Dreams</h1>
            <div class="hero-subtitle">drift through your nocturnal memories</div>
        </div>
    """, unsafe_allow_html=True)

    entries = journal.get_all()

    if not entries:
        st.markdown("*No dreams to explore yet...*")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            search_query = st.text_input("🔍", placeholder="Search your dreams...", label_visibility="collapsed")
        with col2:
            all_tags = set()
            for e in entries:
                all_tags.update(e.tags)
            tag_filter = st.multiselect("Filter by tags", sorted(all_tags), label_visibility="collapsed", placeholder="Filter by tags...")

        filtered = entries
        if search_query:
            filtered = [e for e in filtered if search_query.lower() in e.text.lower()]
        if tag_filter:
            filtered = [e for e in filtered if any(t in e.tags for t in tag_filter)]

        st.markdown(f"<p style='color: #6b5b8a; font-size: 0.85rem;'>Showing {len(filtered)} of {len(entries)} dreams</p>", unsafe_allow_html=True)

        for idx, entry in enumerate(filtered):
            emo_icon = "🌙"
            if entry.emotions:
                top_emo = max(entry.emotions.items(), key=lambda x: x[1])[0]
                emo_map = {
                    "joy": "☀️", "fear": "👁️", "anxiety": "🌊", "sadness": "🌧️",
                    "anger": "🔥", "wonder": "✨", "confusion": "🌀", "peace": "🍃",
                    "love": "💜", "power": "⚡",
                }
                emo_icon = emo_map.get(top_emo, "🌙")

            with st.expander(f"{emo_icon}  {entry.date}  ·  {entry.text[:70]}..."):
                st.markdown(f"*{entry.text}*")

                if entry.tags:
                    tags_html = " ".join(f'<span class="symbol-pill">{t}</span>' for t in entry.tags)
                    st.markdown(tags_html, unsafe_allow_html=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    if entry.symbols:
                        pills = " ".join(f'<span class="symbol-pill">◈ {s}</span>' for s in entry.symbols[:8])
                        st.markdown(f"**Symbols**<br>{pills}", unsafe_allow_html=True)
                with col_b:
                    if entry.emotions:
                        top3 = sorted(entry.emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                        badges = " ".join(
                            f'<span class="emotion-badge">{e.capitalize()} {s:.0%}</span>'
                            for e, s in top3
                        )
                        st.markdown(f"**Emotions**<br>{badges}", unsafe_allow_html=True)

                col_x, col_y = st.columns([1, 1])
                with col_x:
                    if entry.embedding:
                        if st.button("Find echoes ✦", key=f"sim_{idx}_{entry.id}"):
                            similar = find_similar_dreams(entry, journal.entries)
                            if similar:
                                for s in similar:
                                    st.markdown(
                                        f'<div class="dream-card">'
                                        f'<small>{s["date"]} · {s["similarity"]:.0%} resonance</small><br>'
                                        f'{s["preview"]}'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )
                            else:
                                st.caption("No echoes found...")
                with col_y:
                    if st.button("🗑️ Release", key=f"del_{idx}_{entry.id}"):
                        journal.delete_dream(entry.id)
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# ✦ INSIGHTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "✦ Insights":
    st.markdown("""
        <div class="hero">
            <h1>Dream Insights</h1>
            <div class="hero-subtitle">what your dreams are trying to tell you</div>
        </div>
    """, unsafe_allow_html=True)

    entries = journal.get_all()

    if not entries:
        st.markdown("""
            <div style="text-align: center; padding: 40px; color: #6b5b8a; font-style: italic;">
                The void holds no secrets yet... Log some dreams first.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("### ✦ Pattern Analysis")
        insights = generate_insights(entries)
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### ✦ Recurring Dream Patterns")
        recurring = find_recurring_dreams(entries, threshold=0.65)
        if recurring:
            for r in recurring:
                st.markdown(
                    f'<div class="dream-card">'
                    f'<small>ECHO DETECTED · {r["similarity"]:.0%} resonance</small><br><br>'
                    f'<b>{r["dream_a_date"]}</b><br>{r["dream_a_preview"]}...<br><br>'
                    f'<b>{r["dream_b_date"]}</b><br>{r["dream_b_preview"]}...'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="insight-box">No recurring patterns detected yet. '
                'The patterns will emerge as your journal grows...</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        st.markdown("### ✦ Temporal Rhythms")
        temporal = temporal_patterns(entries)
        if temporal["symbols_by_day"]:
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            for day in days_order:
                if day in temporal["symbols_by_day"]:
                    syms = temporal["symbols_by_day"][day]
                    pills = " ".join(f'<span class="symbol-pill">{s} ({c})</span>' for s, c in syms)
                    st.markdown(f"**{day}**<br>{pills}", unsafe_allow_html=True)
                    st.markdown("")
        else:
            st.markdown(
                '<div class="insight-box">Not enough data for temporal patterns. '
                'Keep dreaming...</div>',
                unsafe_allow_html=True,
            )