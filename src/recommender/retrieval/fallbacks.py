"""Explicit routing rules for collaborative and cold-start retrieval."""

from __future__ import annotations

from enum import StrEnum


class RetrievalStrategy(StrEnum):
    TWO_TOWER = "two_tower"
    POPULARITY = "popularity"
    CONTENT = "content"
    CONTENT_POPULARITY = "content_popularity"


def choose_retrieval_strategy(
    *, is_new_user: bool, is_new_item: bool
) -> RetrievalStrategy:
    """Choose a retrieval path without sending unknown IDs to learned towers."""
    if is_new_user and is_new_item:
        return RetrievalStrategy.CONTENT_POPULARITY
    if is_new_user:
        return RetrievalStrategy.POPULARITY
    if is_new_item:
        return RetrievalStrategy.CONTENT
    return RetrievalStrategy.TWO_TOWER
