"""Canonical data models and recommendation logic.

There is exactly ONE representation of a song (`Song`) and ONE representation of
a listener's taste (`UserProfile`). The CSV loader produces `Song` objects, the
scorer consumes them, and the ranker returns them. The application and the tests
travel the same road.

Previously this module had two parallel worlds: `load_songs` returned plain
dicts for the application, while `Song`/`UserProfile`/`Recommender` served the
tests, bridged by an adapter that flattened objects back into dicts. That
adapter silently dropped three fields (valence, danceability, tempo_bpm), so
scoring could never use them without edits in two places. It is now gone.
"""

import csv
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Song:
    """One track in the catalog.

    This dataclass is the single source of truth for what a song *is*. Every
    field the system knows about is declared here, so adding a new attribute is
    a one-line change in one file rather than a hunt through the codebase.

    Field meanings:
        id           unique whole number identifying the track
        title        song name
        artist       performer name
        genre        style label, e.g. "pop" (exact-match only for now)
        mood         feeling label, e.g. "happy"
        energy       0.0 (calm) to 1.0 (intense)
        tempo_bpm    speed in beats per minute
        valence      0.0 (sad sounding) to 1.0 (happy sounding)
        danceability 0.0 to 1.0
        acousticness 0.0 (fully electronic) to 1.0 (fully acoustic)
    """

    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """A listener's stated taste preferences.

    Every field defaults to None, which means "the listener did not say."
    That is deliberately different from stating a value:

        UserProfile(favorite_genre="pop")                       -> acoustic rule skipped
        UserProfile(favorite_genre="pop", likes_acoustic=False) -> acoustic rule applies

    The second profile earns +1.0 on every non-acoustic song, because the
    acoustic rule rewards *agreement* in both directions. The first earns
    nothing from that rule at all. Collapsing "unstated" into "False" would
    silently inflate scores, so the distinction is preserved.
    """

    favorite_genre: Optional[str] = None
    favorite_mood: Optional[str] = None
    target_energy: Optional[float] = None
    likes_acoustic: Optional[bool] = None


def load_songs(csv_path: str) -> List[Song]:
    """Read the song catalog from a CSV file into a list of Song objects.

    A CSV stores everything as text: the file literally contains the characters
    "0", ".", "8", "2". We convert the numeric columns to real numbers here, at
    the boundary where data enters the system, so the rest of the code can do
    arithmetic without worrying about types.

    Each field is now named explicitly rather than looped over generically. That
    is slightly more typing, but it means the loader must satisfy the schema
    declared on `Song` — a column renamed in the CSV fails here, at load time,
    instead of producing a confusing error deep inside scoring.
    """
    songs: List[Song] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append(
                Song(
                    id=int(row["id"]),
                    title=row["title"],
                    artist=row["artist"],
                    genre=row["genre"],
                    mood=row["mood"],
                    energy=float(row["energy"]),
                    tempo_bpm=float(row["tempo_bpm"]),
                    valence=float(row["valence"]),
                    danceability=float(row["danceability"]),
                    acousticness=float(row["acousticness"]),
                )
            )
    return songs


def score_song(user: UserProfile, song: Song) -> Tuple[float, List[str]]:
    """Score one song against one listener profile.

    Returns the total score and a list of human-readable reasons. Every reason
    corresponds to points actually awarded, so an explanation can never claim
    something the scorer did not do.

    The recipe uses three different kinds of comparison:

        mood, genre   exact match, all-or-nothing
        energy        graded by closeness to the target, on a sliding scale
        acoustic      a continuous value thresholded into yes/no, then compared

    Maximum possible score is 8.0 (3.0 + 2.0 + 2.0 + 1.0), reachable only when
    the listener states all four preferences and the song matches all of them.
    """
    score = 0.0
    reasons: List[str] = []

    # Categorical matches: an exact string match earns fixed points.
    if user.favorite_mood is not None and song.mood == user.favorite_mood:
        score += 3.0
        reasons.append(f"mood match: {song.mood} (+3.0)")
    if user.favorite_genre is not None and song.genre == user.favorite_genre:
        score += 2.0
        reasons.append(f"genre match: {song.genre} (+2.0)")

    # Numeric closeness: reward songs whose energy is NEAR the target, not just
    # high. Both values sit in 0.0-1.0, so closeness lands in 0.0-1.0 too.
    if user.target_energy is not None:
        closeness = 1 - abs(song.energy - user.target_energy)
        points = 2.0 * closeness
        score += points
        reasons.append(
            f"energy {song.energy} near target {user.target_energy} (+{points:.2f})"
        )

    # Optional acoustic preference: rewards agreement, either direction.
    if user.likes_acoustic is not None:
        is_acoustic = song.acousticness > 0.5
        if is_acoustic == user.likes_acoustic:
            score += 1.0
            reasons.append("acoustic preference match (+1.0)")

    return score, reasons


def recommend_songs(
    user: UserProfile, songs: List[Song], k: int = 5
) -> List[Tuple[Song, float, str]]:
    """Score every song, then return the top k as (song, score, explanation).

    This is the one and only ranking implementation. Scoring grades a single
    song; ranking is what turns those grades into a recommendation.

    `sorted`/`.sort()` build a new list, so the caller's `songs` list is never
    reordered. Python's sort is *stable*: songs with equal scores keep their
    original catalog order, which is currently our only tie-breaker.
    """
    scored: List[Tuple[Song, float, str]] = []
    for song in songs:
        score, reasons = score_song(user, song)
        explanation = "; ".join(reasons) if reasons else "no strong matches"
        scored.append((song, score, explanation))

    scored.sort(key=lambda item: item[1], reverse=True)  # item[1] is the score
    return scored[:k]


class Recommender:
    """Convenience wrapper that holds a catalog and delegates to the functions.

    This class no longer contains any scoring or ranking logic of its own — it
    forwards to `recommend_songs` and `score_song`. Keeping it is a deliberate
    trade-off: it is a thin, harmless facade that preserves a familiar
    object-oriented entry point, and removing it would churn code for no gain
    right now. Once VibeFlow has a real orchestrator, this class is a candidate
    for deletion.
    """

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top k Songs for this user, ranked highest score first."""
        return [song for song, _score, _explanation in recommend_songs(user, self.songs, k)]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable string explaining a song's score."""
        score, reasons = score_song(user, song)
        detail = "; ".join(reasons) if reasons else "no strong matches"
        return f"Score {score:.2f} — {detail}"
