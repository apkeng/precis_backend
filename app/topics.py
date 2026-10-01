"""The 14 syllabus topics Précis covers, grouped by exam paper.

Ids are stable API values shared with the frontend; anything a clipping is
about that falls outside these topics is not ingested into an event.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Topic:
    id: str
    name: str
    short: str
    group: str
    scope: str  # what the topic covers - shown to Claude when tagging


TOPICS: list[Topic] = [
    Topic("natintl", "National & International Events", "Nat. & Intl.", "Prelims",
          "Major political, economic and cultural happenings."),
    Topic("econsoc", "Economic & Social Development", "Econ. & Social Dev.", "Prelims",
          "Sustainable development, poverty, demographics and inclusion initiatives."),
    Topic("ecology", "Environmental Ecology", "Ecology", "Prelims",
          "Biodiversity, climate change and global conventions."),
    Topic("polgov", "Indian Polity & Governance", "Polity & Gov.", "Prelims",
          "Recent legislation, constitutional amendments and Supreme Court verdicts."),
    Topic("gensci", "General Science", "Gen. Science", "Prelims",
          "Space tech, biotechnology and health-related breakthroughs."),
    Topic("society", "Indian Society", "Society", "GS I",
          "Urbanization, regionalism, secularism and women's issues."),
    Topic("geo", "Geography", "Geography", "GS I",
          "Geophysical phenomena (cyclones, earthquakes) and resource distribution shifts."),
    Topic("const", "Polity & Constitution", "Constitution", "GS II",
          "Landmark judicial rulings, federal tensions and institutional reforms."),
    Topic("socjust", "Social Justice", "Social Justice", "GS II",
          "Health, education and poverty alleviation welfare schemes."),
    Topic("ir", "International Relations", "IR", "GS II",
          "Bilateral ties, global groupings and India's diaspora."),
    Topic("economy", "Economy", "Economy", "GS III",
          "Budget analysis, infrastructure, agriculture and employment trends."),
    Topic("scitech", "Science & Tech", "Sci. & Tech", "GS III",
          "Indigenization of tech, IT and space advancements."),
    Topic("envsec", "Environment & Security", "Env. & Security", "GS III",
          "Pollution mitigation, internal security challenges and border management."),
    Topic("cases", "Case Studies", "Case Studies", "GS IV",
          "Real-world ethical dilemmas, corporate governance and civil service integrity."),
]

TOPIC_IDS: list[str] = [t.id for t in TOPICS]
TOPICS_BY_ID: dict[str, Topic] = {t.id: t for t in TOPICS}

QUESTION_FORMATS: list[str] = ["MCQ", "Statement", "Assertion–Reason", "Match"]
DIFFICULTIES: list[str] = ["Easy", "Mixed", "Hard"]


def describe_topics() -> str:
    """Topic list as prompt text: one `id - name: scope` line per topic."""
    return "\n".join(f"{t.id} - {t.name}: {t.scope}" for t in TOPICS)
