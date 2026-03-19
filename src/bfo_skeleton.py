"""
BFO (Basic Formal Ontology) Skeleton — Top-Level Ontology Anchoring.

Provides a minimal BFO 2.0 category tree that is injected into T-Box prompts
so the LLM anchors every domain class under a well-defined upper category.
This prevents the "generate from zero" categorical errors documented in the
ontology-learning literature.

Reference: BFO 2.0 — https://basic-formal-ontology.org/
"""

from typing import Dict, List

# ────────────────────────────────────────────────────────────────────
# Full BFO 2.0 category hierarchy (pruned to practically useful nodes)
# ────────────────────────────────────────────────────────────────────

BFO_HIERARCHY: Dict[str, dict] = {
    "bfo:Entity": {
        "description": "The root category. Everything is an Entity.",
        "children": ["bfo:Continuant", "bfo:Occurrent"],
    },
    # ── Continuant branch (things that persist) ──
    "bfo:Continuant": {
        "description": "An entity that persists through time while maintaining its identity.",
        "children": [
            "bfo:IndependentContinuant",
            "bfo:SpecificallyDependentContinuant",
            "bfo:GenericallyDependentContinuant",
        ],
    },
    "bfo:IndependentContinuant": {
        "description": "A continuant that is the bearer of qualities and can exist on its own.",
        "children": ["bfo:MaterialEntity", "bfo:ImmaterialEntity"],
    },
    "bfo:MaterialEntity": {
        "description": "An independent continuant made of matter (organisms, artifacts, molecules, substances).",
        "children": [],
    },
    "bfo:ImmaterialEntity": {
        "description": "An independent continuant with no material parts (boundaries, sites, spatial regions).",
        "children": ["bfo:Site", "bfo:SpatialRegion"],
    },
    "bfo:Site": {
        "description": "A three-dimensional immaterial entity bounded by a material entity (e.g. a cavity, a room).",
        "children": [],
    },
    "bfo:SpatialRegion": {
        "description": "A continuant that is a region of space (geographic areas, coordinates).",
        "children": [],
    },
    "bfo:SpecificallyDependentContinuant": {
        "description": "A continuant that depends on a specific independent continuant for its existence.",
        "children": ["bfo:Quality", "bfo:RealizableEntity"],
    },
    "bfo:Quality": {
        "description": "An inherent attribute of an entity (color, mass, temperature, shape).",
        "children": [],
    },
    "bfo:RealizableEntity": {
        "description": "A dependent continuant whose instances can be realized in processes.",
        "children": ["bfo:Role", "bfo:Disposition", "bfo:Function"],
    },
    "bfo:Role": {
        "description": "A realizable entity grounded in social/institutional context (teacher, patient, customer, author).",
        "children": [],
    },
    "bfo:Disposition": {
        "description": "A realizable entity that is a tendency to act in a certain way (fragility, solubility).",
        "children": [],
    },
    "bfo:Function": {
        "description": "A disposition that is the rationale for which an entity was designed or selected (to pump blood, to cut).",
        "children": [],
    },
    "bfo:GenericallyDependentContinuant": {
        "description": "A continuant that depends on one or more independent continuants as carriers.",
        "children": ["bfo:InformationContentEntity"],
    },
    "bfo:InformationContentEntity": {
        "description": "A generically dependent continuant that is about something (documents, data, records, plans, specifications, software).",
        "children": [],
    },
    # ── Occurrent branch (things that happen) ──
    "bfo:Occurrent": {
        "description": "An entity that unfolds or develops over time.",
        "children": ["bfo:Process", "bfo:TemporalRegion"],
    },
    "bfo:Process": {
        "description": "An occurrent that has temporal parts and depends on at least one material entity (events, activities, reactions, workflows).",
        "children": [],
    },
    "bfo:TemporalRegion": {
        "description": "An occurrent that is a region of time (intervals, instants, periods).",
        "children": [],
    },
}


# ────────────────────────────────────────────────────────────────────
# Leaf categories — the ones domain classes should map to
# ────────────────────────────────────────────────────────────────────

BFO_LEAF_CATEGORIES: List[str] = [
    "bfo:MaterialEntity",
    "bfo:InformationContentEntity",
    "bfo:Process",
    "bfo:Role",
    "bfo:Quality",
    "bfo:Function",
    "bfo:Disposition",
    "bfo:SpatialRegion",
    "bfo:TemporalRegion",
    "bfo:Site",
    "bfo:ImmaterialEntity",
]

ALL_BFO_CATEGORIES: List[str] = list(BFO_HIERARCHY.keys())


# ────────────────────────────────────────────────────────────────────
# Prompt-ready skeleton (injected into LLM prompts)
# ────────────────────────────────────────────────────────────────────

def format_bfo_for_prompt() -> str:
    """Return a compact, LLM-readable BFO skeleton for prompt injection."""
    lines = [
        "Every domain class MUST be anchored under exactly ONE of these BFO leaf categories:",
        "",
    ]
    for cat in BFO_LEAF_CATEGORIES:
        desc = BFO_HIERARCHY[cat]["description"]
        lines.append(f"  - {cat}: {desc}")

    lines.append("")
    lines.append("EXAMPLES of correct anchoring:")
    lines.append("  ex:Document -> bfo:InformationContentEntity")
    lines.append("  ex:Protein  -> bfo:MaterialEntity")
    lines.append("  ex:Author   -> bfo:Role")
    lines.append("  ex:Reaction -> bfo:Process")
    lines.append("  ex:Color    -> bfo:Quality")
    lines.append("  ex:Region   -> bfo:SpatialRegion")
    return "\n".join(lines)


BFO_PROMPT_SKELETON: str = format_bfo_for_prompt()


def is_valid_bfo_parent(uri: str) -> bool:
    """Check whether a URI is a recognized BFO category."""
    return uri in ALL_BFO_CATEGORIES


def get_bfo_lineage(category: str) -> List[str]:
    """Return the path from bfo:Entity down to *category* (inclusive)."""
    # BFS parent lookup
    parent_map: Dict[str, str] = {}
    for cat, info in BFO_HIERARCHY.items():
        for child in info["children"]:
            parent_map[child] = cat

    path = [category]
    current = category
    while current in parent_map:
        current = parent_map[current]
        path.append(current)
    path.reverse()
    return path
