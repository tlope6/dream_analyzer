"""
Dream Journal — storage and retrieval of dream entries.
Uses a local JSON file for simplicity.
"""

import json
import os
import uuid
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from typing import Optional


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
JOURNAL_FILE = os.path.join(DATA_DIR, "dreams.json")


@dataclass
class DreamEntry:
    """A single dream journal entry."""
    id: str
    date: str                          # ISO format date
    text: str                          # Raw dream description
    tags: list[str] = field(default_factory=list)
    created_at: str = ""

    # --- Populated by NLP pipeline ---
    symbols: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    emotions: dict = field(default_factory=dict)      # {label: score}
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DreamEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class DreamJournal:
    """Manages dream entries — load, save, search."""

    def __init__(self, filepath: str = JOURNAL_FILE):
        self.filepath = filepath
        self.entries: list[DreamEntry] = []
        self._ensure_data_dir()
        self.load()

    def _ensure_data_dir(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def load(self):
        """Load entries from JSON file."""
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                data = json.load(f)
            self.entries = [DreamEntry.from_dict(d) for d in data]
        else:
            self.entries = []

    def save(self):
        """Persist all entries to JSON file."""
        with open(self.filepath, "w") as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2)

    def add_dream(
        self,
        text: str,
        dream_date: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> DreamEntry:
        """Add a new dream entry and save."""
        entry = DreamEntry(
            id=str(uuid.uuid4())[:8],
            date=dream_date or date.today().isoformat(),
            text=text,
            tags=tags or [],
            created_at=datetime.now().isoformat(),
        )
        self.entries.append(entry)
        self.save()
        return entry

    def delete_dream(self, dream_id: str) -> bool:
        """Delete a dream by ID."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e.id != dream_id]
        if len(self.entries) < before:
            self.save()
            return True
        return False

    def get_dream(self, dream_id: str) -> Optional[DreamEntry]:
        """Get a single dream by ID."""
        for e in self.entries:
            if e.id == dream_id:
                return e
        return None

    def get_all(self) -> list[DreamEntry]:
        """Return all entries sorted by date (newest first)."""
        return sorted(self.entries, key=lambda e: e.date, reverse=True)

    def get_by_date_range(self, start: str, end: str) -> list[DreamEntry]:
        """Filter entries within a date range (inclusive)."""
        return [e for e in self.entries if start <= e.date <= end]

    def get_by_tag(self, tag: str) -> list[DreamEntry]:
        """Filter entries that have a specific tag."""
        return [e for e in self.entries if tag.lower() in [t.lower() for t in e.tags]]

    def search_text(self, query: str) -> list[DreamEntry]:
        """Simple text search across dream entries."""
        query_lower = query.lower()
        return [e for e in self.entries if query_lower in e.text.lower()]

    def update_entry(self, entry: DreamEntry):
        """Update an existing entry in-place and save."""
        for i, e in enumerate(self.entries):
            if e.id == entry.id:
                self.entries[i] = entry
                self.save()
                return
        raise ValueError(f"Entry {entry.id} not found")

    def get_sample_dreams(self) -> list[dict]:
        """Return sample dreams for demo/testing purposes."""
        return [
            {
                "text": "I was flying over a vast ocean at sunset. The water was crystal clear and I could see fish swimming below. Suddenly I started falling and woke up just before hitting the water.",
                "date": "2025-01-15",
                "tags": ["flying", "lucid"],
            },
            {
                "text": "I was back in my old high school taking an exam I hadn't studied for. The classroom was dark and all the other students were strangers. I couldn't read any of the questions on the paper.",
                "date": "2025-01-18",
                "tags": ["school", "nightmare"],
            },
            {
                "text": "Walking through a beautiful forest with my grandmother who passed away years ago. She was showing me different flowers and telling me their names. I felt incredibly peaceful.",
                "date": "2025-01-22",
                "tags": ["nature", "deceased_relative"],
            },
            {
                "text": "I was being chased through a dark city by a shadowy figure. Every door I tried was locked. I found a key in my pocket and opened a door that led to a bright garden.",
                "date": "2025-01-28",
                "tags": ["nightmare", "recurring"],
            },
            {
                "text": "I was swimming in a river that kept getting wider and deeper. A giant snake appeared in the water but it was friendly and guided me to shore. On the shore there was a house I'd never seen but felt like home.",
                "date": "2025-02-01",
                "tags": ["water", "animals"],
            },
            {
                "text": "My teeth started falling out one by one during a dinner party. Nobody else noticed. I kept trying to put them back in. Then I looked in a mirror and my reflection was smiling with perfect teeth.",
                "date": "2025-02-05",
                "tags": ["recurring", "nightmare"],
            },
            {
                "text": "I was on a train traveling through mountains. The train had no driver. I went to the front car and found a map that showed the route going in a perfect circle. Other passengers were reading books with blank pages.",
                "date": "2025-02-10",
                "tags": ["travel"],
            },
            {
                "text": "I was in a huge library where the books were alive and whispering. I picked up one and it started telling me secrets about my future. A bird flew in through a window and took the book away.",
                "date": "2025-02-14",
                "tags": ["surreal"],
            },
            {
                "text": "Standing on top of a mountain during a thunderstorm. Lightning kept striking all around me but I wasn't afraid. I felt powerful. Then rain started and the mountain slowly dissolved into sand beneath my feet.",
                "date": "2025-02-18",
                "tags": ["nature", "lucid"],
            },
            {
                "text": "I was at a wedding but couldn't figure out who was getting married. Everyone was wearing masks. When I took off my mask, I realized I was the one getting married to a stranger. The ocean was visible through the church windows.",
                "date": "2025-02-22",
                "tags": ["social", "surreal"],
            },
        ]