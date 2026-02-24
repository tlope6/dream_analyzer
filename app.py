"""
Dream Journal Analyzer — Streamlit Dashboard
=============================================
Run with: streamlit run app.py
"""

import streamlit as st
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from journal import DreamJournal
from np_pipeline import process_dream, USE_TRANSFORMERS
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
    page_title="Dream Journal Analyzer 🌙",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0e0e1a; }
    .dream-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    .insight-box {
        background: linear-gradient(135deg, #1e1e3a 0%, #2d1b4e 100%);
        border-left: 4px solid #9370DB;
        padding: 15px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    .stat-card {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialize journal ─────────────────────────────────────────────────────
@st.cache_resource
def get_journal():
    return DreamJournal()


journal = get_journal()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌙 Dream Journal")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📝 Log a Dream", "📊 Dashboard", "🔍 Explore Dreams", "💡 Insights"],
        index=1,
    )

    st.markdown("---")
    st.caption(f"📚 {len(journal.entries)} dreams logged")

    if USE_TRANSFORMERS:
        st.caption("🧠 AI models: enabled")
    else:
        st.caption("⚡ Lightweight mode (no transformers)")

    st.markdown("---")

    # Load sample data button
    if st.button("📦 Load Sample Dreams", use_container_width=True):
        samples = journal.get_sample_dreams()
        count = 0
        for sample in samples:
            # Check if dream with same date+text already exists
            existing = [e for e in journal.entries if e.date == sample["date"]]
            if not existing:
                entry = journal.add_dream(
                    text=sample["text"],
                    dream_date=sample["date"],
                    tags=sample.get("tags", []),
                )
                # Process with NLP
                result = process_dream(entry.text)
                entry.symbols = result["symbol_names"]
                entry.entities = result["entities"]
                entry.emotions = result["emotions"]
                entry.embedding = result["embedding"]
                journal.update_entry(entry)
                count += 1
        if count > 0:
            st.success(f"Loaded {count} sample dreams!")
            st.rerun()
        else:
            st.info("Sample dreams already loaded!")


# ══════════════════════════════════════════════════════════════════════════
# 📝 LOG A DREAM
# ══════════════════════════════════════════════════════════════════════════
if page == "📝 Log a Dream":
    st.header("📝 Log a New Dream")
    st.markdown("*Record your dream while it's still fresh...*")

    col1, col2 = st.columns([3, 1])

    with col1:
        dream_text = st.text_area(
            "Describe your dream",
            height=200,
            placeholder="Last night I dreamed that I was flying over an endless ocean...",
        )

    with col2:
        dream_date = st.date_input("Date")
        tags_input = st.text_input(
            "Tags (comma-separated)",
            placeholder="lucid, recurring, nightmare",
        )

    if st.button("🌙 Save & Analyze", type="primary", use_container_width=True):
        if dream_text.strip():
            tags = [t.strip() for t in tags_input.split(",") if t.strip()] if tags_input else []

            with st.spinner("Analyzing your dream..."):
                # Save entry
                entry = journal.add_dream(
                    text=dream_text,
                    dream_date=dream_date.isoformat(),
                    tags=tags,
                )

                # Process with NLP pipeline
                result = process_dream(dream_text)
                entry.symbols = result["symbol_names"]
                entry.entities = result["entities"]
                entry.emotions = result["emotions"]
                entry.embedding = result["embedding"]
                journal.update_entry(entry)

            st.success("Dream saved and analyzed! ✨")

            # Show analysis results
            st.markdown("### Analysis Results")

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**🔮 Symbols Found**")
                if result["symbols"]:
                    for sym in result["symbols"]:
                        source_icon = {"dream_dictionary": "📖", "spacy_ner": "🏷️", "spacy_pos": "📌"}.get(sym["source"], "•")
                        st.markdown(f"{source_icon} **{sym['text']}** — {sym['label']}")
                else:
                    st.info("No specific symbols detected.")

            with col_b:
                st.markdown("**😊 Emotional Tone**")
                if result["emotions"]:
                    sorted_emos = sorted(result["emotions"].items(), key=lambda x: x[1], reverse=True)
                    for emo, score in sorted_emos[:5]:
                        bar_width = int(score * 100)
                        st.markdown(f"**{emo.capitalize()}**: {score:.1%}")
                        st.progress(score)

            # Similar dreams
            if entry.embedding:
                similar = find_similar_dreams(entry, journal.entries, top_n=3)
                if similar:
                    st.markdown("### 🔗 Similar Past Dreams")
                    for s in similar:
                        st.markdown(
                            f'<div class="dream-card">'
                            f'<small>{s["date"]} · Similarity: {s["similarity"]:.0%}</small><br>'
                            f'{s["preview"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
        else:
            st.warning("Please describe your dream first!")


# ══════════════════════════════════════════════════════════════════════════
# 📊 DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.header("📊 Dream Dashboard")

    entries = journal.get_all()

    if not entries:
        st.info("No dreams logged yet. Go to '📝 Log a Dream' to get started, or click '📦 Load Sample Dreams' in the sidebar!")
    else:
        # Stats row
        col1, col2, col3, col4 = st.columns(4)
        freq = symbol_frequency(entries)
        with col1:
            st.metric("Total Dreams", len(entries))
        with col2:
            st.metric("Unique Symbols", len(freq))
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

        # Charts
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Emotion Timeline",
            "🔮 Symbols",
            "🕸️ Connections",
            "☁️ Word Cloud",
        ])

        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(emotion_timeline_chart(entries), use_container_width=True)
            with col_b:
                st.plotly_chart(emotion_radar_chart(entries), use_container_width=True)

            st.plotly_chart(dominant_emotion_pie(entries), use_container_width=True)

        with tab2:
            st.plotly_chart(symbol_bar_chart(entries), use_container_width=True)

        with tab3:
            st.plotly_chart(co_occurrence_network(entries, min_count=1), use_container_width=True)

            # Co-occurrence table
            co_occ = symbol_co_occurrence(entries, min_count=1)
            if co_occ:
                st.markdown("**Symbol Pairs That Appear Together**")
                for pair in co_occ[:10]:
                    st.markdown(
                        f"🔗 **{pair['symbol_a']}** ↔ **{pair['symbol_b']}** "
                        f"({pair['count']} dreams)"
                    )

        with tab4:
            wc_fig = dream_word_cloud(entries)
            if wc_fig:
                st.pyplot(wc_fig)
            else:
                st.info("Not enough data for word cloud yet.")


# ══════════════════════════════════════════════════════════════════════════
# 🔍 EXPLORE DREAMS
# ══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Explore Dreams":
    st.header("🔍 Explore Your Dreams")

    entries = journal.get_all()

    if not entries:
        st.info("No dreams to explore yet!")
    else:
        # Search & filter
        col1, col2 = st.columns([2, 1])
        with col1:
            search_query = st.text_input("🔍 Search dreams", placeholder="water, flying, school...")
        with col2:
            all_tags = set()
            for e in entries:
                all_tags.update(e.tags)
            tag_filter = st.multiselect("Filter by tags", sorted(all_tags))

        # Apply filters
        filtered = entries
        if search_query:
            filtered = [e for e in filtered if search_query.lower() in e.text.lower()]
        if tag_filter:
            filtered = [e for e in filtered if any(t in e.tags for t in tag_filter)]

        st.caption(f"Showing {len(filtered)} of {len(entries)} dreams")

        # Display dreams
        for entry in filtered:
            with st.expander(f"🌙 {entry.date}  |  {entry.text[:80]}..."):
                st.markdown(entry.text)

                if entry.tags:
                    st.markdown("**Tags:** " + ", ".join(f"`{t}`" for t in entry.tags))

                col_a, col_b = st.columns(2)

                with col_a:
                    if entry.symbols:
                        st.markdown("**🔮 Symbols:** " + ", ".join(entry.symbols[:10]))

                with col_b:
                    if entry.emotions:
                        top3 = sorted(entry.emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                        emo_str = " · ".join(f"{e.capitalize()} ({s:.0%})" for e, s in top3)
                        st.markdown(f"**😊 Emotions:** {emo_str}")

                # Similar dreams button
                if entry.embedding:
                    if st.button(f"Find similar dreams", key=f"sim_{entry.id}"):
                        similar = find_similar_dreams(entry, journal.entries)
                        if similar:
                            for s in similar:
                                st.markdown(
                                    f"  ↳ **{s['date']}** ({s['similarity']:.0%} similar) — {s['preview']}"
                                )
                        else:
                            st.caption("No similar dreams found.")

                # Delete button
                if st.button(f"🗑️ Delete", key=f"del_{entry.id}"):
                    journal.delete_dream(entry.id)
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# 💡 INSIGHTS
# ══════════════════════════════════════════════════════════════════════════
elif page == "💡 Insights":
    st.header("💡 Dream Insights")

    entries = journal.get_all()

    if not entries:
        st.info("Log some dreams first to see insights!")
    else:
        # Auto-generated insights
        st.markdown("### 🔮 Pattern Analysis")
        insights = generate_insights(entries)
        for insight in insights:
            st.markdown(
                f'<div class="insight-box">{insight}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Recurring dreams
        st.markdown("### 🔄 Potential Recurring Dreams")
        recurring = find_recurring_dreams(entries, threshold=0.65)
        if recurring:
            for r in recurring:
                st.markdown(
                    f'<div class="dream-card">'
                    f'<b>{r["dream_a_date"]}</b>: {r["dream_a_preview"]}...<br>'
                    f'<b>{r["dream_b_date"]}</b>: {r["dream_b_preview"]}...<br>'
                    f'<small>Similarity: {r["similarity"]:.0%}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No recurring dream patterns detected yet. Keep logging!")

        st.markdown("---")

        # Temporal patterns
        st.markdown("### 📅 Day-of-Week Patterns")
        temporal = temporal_patterns(entries)
        if temporal["symbols_by_day"]:
            for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
                if day in temporal["symbols_by_day"]:
                    syms = temporal["symbols_by_day"][day]
                    sym_str = ", ".join(f"{s} ({c})" for s, c in syms)
                    st.markdown(f"**{day}**: {sym_str}")
        else:
            st.caption("Not enough data for day-of-week patterns yet.")