"""
Visualizations — generates Plotly charts and other visuals for the dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from collections import Counter
from io import BytesIO

from journal import DreamEntry
from analyzer import (
    symbol_frequency,
    symbol_co_occurrence,
    emotion_timeline,
    dominant_emotion_per_dream,
)


# ── Color palette ──────────────────────────────────────────────────────────
EMOTION_COLORS = {
    "joy": "#FFD700",
    "fear": "#8B0000",
    "anxiety": "#FF6347",
    "sadness": "#4682B4",
    "anger": "#DC143C",
    "wonder": "#9370DB",
    "confusion": "#A9A9A9",
    "peace": "#90EE90",
    "love": "#FF69B4",
    "power": "#FF8C00",
}


def emotion_timeline_chart(entries: list[DreamEntry]) -> go.Figure:
    """Interactive line chart of emotions over time."""
    timeline = emotion_timeline(entries)
    if not timeline:
        return _empty_figure("No emotion data yet")

    df = pd.DataFrame(timeline)
    emotion_cols = [c for c in df.columns if c not in ("date", "dream_id")]

    fig = go.Figure()
    for emotion in emotion_cols:
        if emotion in df.columns:
            fig.add_trace(go.Scatter(
                x=df["date"],
                y=df[emotion],
                mode="lines+markers",
                name=emotion.capitalize(),
                line=dict(color=EMOTION_COLORS.get(emotion, "#888")),
                hovertemplate=f"{emotion}: %{{y:.2f}}<extra></extra>",
            ))

    fig.update_layout(
        title="Emotional Arc of Your Dreams",
        xaxis_title="Date",
        yaxis_title="Emotion Score",
        yaxis_range=[0, 1],
        template="plotly_dark",
        hovermode="x unified",
        height=400,
    )
    return fig


def emotion_radar_chart(entries: list[DreamEntry]) -> go.Figure:
    """Radar chart showing average emotion profile across all dreams."""
    if not entries:
        return _empty_figure("No dreams yet")

    emotion_sums = Counter()
    count = 0
    for entry in entries:
        if entry.emotions:
            for emo, score in entry.emotions.items():
                emotion_sums[emo] += score
            count += 1

    if count == 0:
        return _empty_figure("No emotion data")

    emotions = list(emotion_sums.keys())
    values = [emotion_sums[e] / count for e in emotions]

    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],  # Close the polygon
        theta=[e.capitalize() for e in emotions] + [emotions[0].capitalize()],
        fill="toself",
        fillcolor="rgba(147, 112, 219, 0.3)",
        line=dict(color="#9370DB"),
    ))

    fig.update_layout(
        title="Your Dream Emotion Profile",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template="plotly_dark",
        height=400,
    )
    return fig


def symbol_bar_chart(entries: list[DreamEntry], top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of most common dream symbols."""
    freq = symbol_frequency(entries)
    if not freq:
        return _empty_figure("No symbols extracted yet")

    most_common = freq.most_common(top_n)
    symbols = [s for s, _ in reversed(most_common)]
    counts = [c for _, c in reversed(most_common)]

    fig = go.Figure(go.Bar(
        x=counts,
        y=symbols,
        orientation="h",
        marker_color="#9370DB",
    ))

    fig.update_layout(
        title=f"Top {top_n} Dream Symbols",
        xaxis_title="Frequency",
        template="plotly_dark",
        height=max(300, top_n * 28),
    )
    return fig


def co_occurrence_network(entries: list[DreamEntry], min_count: int = 2) -> go.Figure:
    """Network graph showing symbol co-occurrence relationships."""
    import networkx as nx

    co_occ = symbol_co_occurrence(entries, min_count=min_count)
    if not co_occ:
        return _empty_figure("Not enough co-occurring symbols yet")

    G = nx.Graph()
    for pair in co_occ:
        G.add_edge(pair["symbol_a"], pair["symbol_b"], weight=pair["count"])

    pos = nx.spring_layout(G, seed=42, k=2)

    # Edges
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="#555"),
        hoverinfo="none",
        mode="lines",
    )

    # Nodes
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_sizes = [max(15, G.degree(n) * 8) for n in G.nodes()]
    node_text = [
        f"{n}<br>Connections: {G.degree(n)}"
        for n in G.nodes()
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=node_text,
        text=list(G.nodes()),
        textposition="top center",
        textfont=dict(size=10, color="white"),
        marker=dict(
            size=node_sizes,
            color="#9370DB",
            line=dict(width=1, color="white"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Symbol Connection Network",
        showlegend=False,
        template="plotly_dark",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
    )
    return fig


def dream_word_cloud(entries: list[DreamEntry]):
    """
    Generate a word cloud image from dream symbols.
    Returns a matplotlib figure (for st.pyplot).
    """
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    freq = symbol_frequency(entries)
    if not freq:
        return None

    wc = WordCloud(
        width=800,
        height=400,
        background_color="#1a1a2e",
        colormap="twilight",
        max_words=80,
        prefer_horizontal=0.7,
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor("#1a1a2e")
    return fig


def dominant_emotion_pie(entries: list[DreamEntry]) -> go.Figure:
    """Pie chart showing distribution of dominant emotions."""
    dom = dominant_emotion_per_dream(entries)
    if not dom:
        return _empty_figure("No emotion data")

    emotion_counts = Counter(d["emotion"] for d in dom)
    labels = list(emotion_counts.keys())
    values = list(emotion_counts.values())
    colors = [EMOTION_COLORS.get(e, "#888") for e in labels]

    fig = go.Figure(go.Pie(
        labels=[e.capitalize() for e in labels],
        values=values,
        marker=dict(colors=colors),
        hole=0.4,
    ))

    fig.update_layout(
        title="Dominant Emotions Distribution",
        template="plotly_dark",
        height=400,
    )
    return fig


def _empty_figure(message: str) -> go.Figure:
    """Return an empty placeholder figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        template="plotly_dark",
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig