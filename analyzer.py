"""
Dream Analyzer — finds patterns, trends, and connections across dream entries.
"""

import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from typing import Optional

from journal import DreamEntry


def symbol_frequency(entries: list[DreamEntry]) -> Counter:
    """Count how often each symbol appears across all dreams."""
    counter = Counter()
    for entry in entries:
        counter.update(entry.symbols)
    return counter


def symbol_co_occurrence(entries: list[DreamEntry], min_count: int = 2) -> list[dict]:
    """
    Find symbols that frequently appear together in the same dream.
    Returns list of {symbol_a, symbol_b, count} sorted by count.
    """
    pair_counts = Counter()
    for entry in entries:
        unique_symbols = list(set(entry.symbols))
        for a, b in combinations(sorted(unique_symbols), 2):
            pair_counts[(a, b)] += 1

    results = [
        {"symbol_a": a, "symbol_b": b, "count": count}
        for (a, b), count in pair_counts.items()
        if count >= min_count
    ]
    return sorted(results, key=lambda x: x["count"], reverse=True)


def emotion_timeline(entries: list[DreamEntry]) -> list[dict]:
    """
    Get emotion scores over time for timeline visualization.
    Returns list of {date, emotion_name: score, ...} sorted by date.
    """
    timeline = []
    for entry in sorted(entries, key=lambda e: e.date):
        if entry.emotions:
            row = {"date": entry.date, "dream_id": entry.id}
            row.update(entry.emotions)
            timeline.append(row)
    return timeline


def dominant_emotion_per_dream(entries: list[DreamEntry]) -> list[dict]:
    """Get the strongest emotion for each dream."""
    results = []
    for entry in entries:
        if entry.emotions:
            top_emotion = max(entry.emotions.keys(), key=lambda e: entry.emotions[e])
            results.append({
                "date": entry.date,
                "dream_id": entry.id,
                "emotion": top_emotion,
                "score": entry.emotions[top_emotion],
                "preview": entry.text[:80] + "...",
            })
    return results


def temporal_patterns(entries: list[DreamEntry]) -> dict:
    """
    Analyze when certain symbols/emotions appear.
    Groups by day of week.
    """
    from datetime import datetime

    day_symbols = defaultdict(list)
    day_emotions = defaultdict(list)

    for entry in entries:
        try:
            dt = datetime.fromisoformat(entry.date)
            day_name = dt.strftime("%A")
        except ValueError:
            continue

        day_symbols[day_name].extend(entry.symbols)

        if entry.emotions:
            top = max(entry.emotions.keys(), key=lambda e: entry.emotions[e])
            day_emotions[day_name].append(top)

    # Most common symbol per day
    day_top_symbols = {}
    for day, syms in day_symbols.items():
        if syms:
            day_top_symbols[day] = Counter(syms).most_common(3)

    day_top_emotions = {}
    for day, emos in day_emotions.items():
        if emos:
            day_top_emotions[day] = Counter(emos).most_common(3)

    return {
        "symbols_by_day": day_top_symbols,
        "emotions_by_day": day_top_emotions,
    }


def find_similar_dreams(
    target: DreamEntry,
    entries: list[DreamEntry],
    top_n: int = 5,
) -> list[dict]:
    """
    Find dreams most similar to a target dream using cosine similarity on embeddings.
    """
    if not target.embedding:
        return []

    target_vec = np.array(target.embedding)
    results = []

    for entry in entries:
        if entry.id == target.id or not entry.embedding:
            continue
        entry_vec = np.array(entry.embedding)
        # Cosine similarity
        dot = np.dot(target_vec, entry_vec)
        norm = np.linalg.norm(target_vec) * np.linalg.norm(entry_vec)
        similarity = float(dot / norm) if norm > 0 else 0.0

        results.append({
            "dream_id": entry.id,
            "date": entry.date,
            "similarity": round(similarity, 3),
            "preview": entry.text[:100] + "...",
        })

    return sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_n]


def find_recurring_dreams(entries: list[DreamEntry], threshold: float = 0.75) -> list[dict]:
    """
    Detect recurring dreams — pairs of dreams with high semantic similarity.
    """
    recurring = []
    checked = set()

    for i, entry_a in enumerate(entries):
        if not entry_a.embedding:
            continue
        vec_a = np.array(entry_a.embedding)

        for j, entry_b in enumerate(entries):
            if j <= i or not entry_b.embedding:
                continue
            pair_key = (entry_a.id, entry_b.id)
            if pair_key in checked:
                continue
            checked.add(pair_key)

            vec_b = np.array(entry_b.embedding)
            dot = np.dot(vec_a, vec_b)
            norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
            sim = float(dot / norm) if norm > 0 else 0.0

            if sim >= threshold:
                recurring.append({
                    "dream_a_id": entry_a.id,
                    "dream_a_date": entry_a.date,
                    "dream_a_preview": entry_a.text[:80],
                    "dream_b_id": entry_b.id,
                    "dream_b_date": entry_b.date,
                    "dream_b_preview": entry_b.text[:80],
                    "similarity": round(sim, 3),
                })

    return sorted(recurring, key=lambda x: x["similarity"], reverse=True)


def generate_insights(entries: list[DreamEntry]) -> list[str]:
    """
    Generate human-readable insights from the dream data.
    Returns a list of insight strings.
    """
    if not entries:
        return ["No dreams recorded yet. Start logging your dreams!"]

    insights = []

    # Most common symbols
    freq = symbol_frequency(entries)
    if freq:
        top3 = freq.most_common(3)
        symbols_str = ", ".join(f'"{s}"' for s, _ in top3)
        insights.append(f"Your most recurring dream symbols are {symbols_str}.")

    # Dominant emotions
    emotion_totals = Counter()
    for entry in entries:
        if entry.emotions:
            top = max(entry.emotions, key=lambda e: entry.emotions[e])
            emotion_totals[top] += 1
    if emotion_totals:
        top_emotion = emotion_totals.most_common(1)[0][0]
        insights.append(
            f"The dominant emotion across your dreams is **{top_emotion}** "
            f"(appears in {emotion_totals[top_emotion]} of {len(entries)} dreams)."
        )

    # Co-occurring symbols
    co_occ = symbol_co_occurrence(entries, min_count=2)
    if co_occ:
        top_pair = co_occ[0]
        insights.append(
            f'"{top_pair["symbol_a"]}" and "{top_pair["symbol_b"]}" '
            f'frequently appear together in your dreams ({top_pair["count"]} times).'
        )

    # Temporal patterns
    temporal = temporal_patterns(entries)
    if temporal["emotions_by_day"]:
        for day, emos in temporal["emotions_by_day"].items():
            if emos and emos[0][1] >= 2:
                insights.append(
                    f"You tend to have **{emos[0][0]}** dreams on {day}s."
                )

    if len(insights) == 0:
        insights.append("Keep logging dreams — patterns will emerge over time!")

    return insights