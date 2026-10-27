"""Navigation model for the DataLab Pro application.

This module deliberately contains no analytical logic.  It defines the
product-level workspace hierarchy used by the Streamlit interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class NavigationItem:
    """A single DataLab workspace."""

    key: str
    label: str
    icon: str
    description: str


@dataclass(frozen=True)
class NavigationGroup:
    """A related group of DataLab workspaces."""

    label: str
    items: tuple[NavigationItem, ...]


HOME: Final = NavigationItem(
    key="home",
    label="Home",
    icon="⌂",
    description="Workspace overview and recent activity.",
)


NAVIGATION_GROUPS: Final[tuple[NavigationGroup, ...]] = (
    NavigationGroup(
        label="DATA",
        items=(
            NavigationItem(
                key="data",
                label="Data",
                icon="▦",
                description="Inspect datasets, schemas and data quality.",
            ),
            NavigationItem(
                key="clean",
                label="Clean",
                icon="◇",
                description="Clean missing values, duplicates and text fields.",
            ),
        ),
    ),
    NavigationGroup(
        label="ANALYZE",
        items=(
            NavigationItem(
                key="explore",
                label="Explore",
                icon="◫",
                description="Explore distributions, relationships and patterns.",
            ),
            NavigationItem(
                key="statistics",
                label="Statistical Analysis",
                icon="∑",
                description=(
                    "Run descriptive, inferential and diagnostic "
                    "statistical analyses."
                ),
            ),
        ),
    ),
    NavigationGroup(
        label="BUILD",
        items=(
            NavigationItem(
                key="features",
                label="Feature Engineering",
                icon="⚙",
                description="Prepare and transform predictors.",
            ),
            NavigationItem(
                key="models",
                label="Model Builder",
                icon="◈",
                description="Train and evaluate predictive models.",
            ),
            NavigationItem(
                key="clustering",
                label="Clustering",
                icon="◎",
                description="Discover structure using unsupervised learning.",
            ),
        ),
    ),
    NavigationGroup(
        label="OUTPUT",
        items=(
            NavigationItem(
                key="reports",
                label="Reports",
                icon="▤",
                description="Review analytical results and reports.",
            ),
            NavigationItem(
                key="export",
                label="Export",
                icon="↓",
                description="Export datasets, predictions and results.",
            ),
        ),
    ),
)


def all_navigation_items() -> tuple[NavigationItem, ...]:
    """Return every navigable workspace."""

    items: list[NavigationItem] = [HOME]

    for group in NAVIGATION_GROUPS:
        items.extend(group.items)

    return tuple(items)


def navigation_item(key: str) -> NavigationItem:
    """Return navigation metadata for a workspace key."""

    for item in all_navigation_items():
        if item.key == key:
            return item

    raise KeyError(f"Unknown DataLab workspace: {key!r}")