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
from typing import Dict, Iterator, List, Optional, Tuple

from src.exceptions import CatalogError, InvalidSongError

# Columns the CSV must provide. Extra columns are ignored, which lets the data
# file carry notes without breaking the loader.
REQUIRED_COLUMNS = (
    "id",
    "title",
    "artist",
    "genre",
    "mood",
    "energy",
    "tempo_bpm",
    "valence",
    "danceability",
    "acousticness",
)

# Fields that describe a proportion and must therefore sit inside 0.0-1.0.
# The scoring formula 2.0 * (1 - abs(a - b)) only behaves for values in this
# range; outside it, scores go negative and rankings become nonsense.
UNIT_RANGE_FIELDS = ("energy", "valence", "danceability", "acousticness")


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

    def __post_init__(self) -> None:
        """Validate this song's own invariants the moment it is constructed.

        `__post_init__` is a dataclass hook: it runs automatically right after
        the generated `__init__` has assigned the fields. Putting the checks here
        means a Song that breaks its own rules cannot exist at all — not from a
        CSV, not from test code, not from a future LLM-generated suggestion.

        This is the "make illegal states unrepresentable" idea. Validating only
        at the CSV boundary would still allow Song(energy=5.0) in code, and the
        scoring formula would silently return a negative score for it.

        Type hints do NOT do this for us. Python does not check annotations at
        runtime, so `energy: float` happily accepts the string "loud".
        """
        problems: List[str] = []

        if not isinstance(self.id, int) or isinstance(self.id, bool):
            problems.append(f"id must be a whole number, got {self.id!r}")

        for field_name in ("title", "artist", "genre", "mood"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{field_name} must be a non-empty string, got {value!r}")

        for field_name in UNIT_RANGE_FIELDS:
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"{field_name} must be a number, got {value!r}")
            elif not 0.0 <= value <= 1.0:
                problems.append(f"{field_name} must be between 0.0 and 1.0, got {value}")

        if not isinstance(self.tempo_bpm, (int, float)) or isinstance(self.tempo_bpm, bool):
            problems.append(f"tempo_bpm must be a number, got {self.tempo_bpm!r}")
        elif self.tempo_bpm <= 0:
            problems.append(f"tempo_bpm must be greater than 0, got {self.tempo_bpm}")

        if problems:
            raise InvalidSongError("; ".join(problems))


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

    def __post_init__(self) -> None:
        """Reject a target energy outside 0.0-1.0.

        Validating songs alone would not fully close the negative-score hole,
        because the energy term compares a song against a *target*. A target of
        5.0 against a valid song still yields 2.0 * (1 - 4.5) = -7.0. Both sides
        of the comparison have to be in range for the formula to mean anything.
        """
        if self.target_energy is None:
            return
        if isinstance(self.target_energy, bool) or not isinstance(
            self.target_energy, (int, float)
        ):
            raise InvalidSongError(
                f"target_energy must be a number, got {self.target_energy!r}"
            )
        if not 0.0 <= self.target_energy <= 1.0:
            raise InvalidSongError(
                f"target_energy must be between 0.0 and 1.0, got {self.target_energy}"
            )


@dataclass(frozen=True)
class ScoringWeights:
    """How much each feature contributes to a song's score.

    These numbers were previously literals buried inside `score_song`, which
    meant the only way to try different priorities was to edit the source and
    edit it back. Pulling them out makes a weighting into a value you can name,
    pass around, compare against another, and assert on in a test.

    The defaults reproduce the original hard-coded behavior exactly, so existing
    scores are unchanged.

        mood_match     awarded when the song's mood equals the stated mood
        genre_match    awarded when the song's genre equals the stated genre
        energy_match   the MAXIMUM awarded for energy; the actual award is this
                       value scaled by how close the song is to the target, so a
                       dead-on match earns all of it and a distant song earns
                       almost none
        acoustic_match awarded when the song agrees with the acoustic preference

    `acoustic_threshold` is not a weight — it is the cutoff above which a song
    counts as "acoustic". It lives here because it is the same kind of thing: a
    tuning knob that used to be an unexplained literal (`> 0.5`). Keeping the
    class named ScoringWeights and documenting the exception is clearer than
    inventing a second config object for one value.

    Frozen (immutable) on purpose. A weighting should not change halfway through
    a run, and immutability is what makes it safe to use as a default argument.
    """

    mood_match: float = 3.0
    genre_match: float = 2.0
    energy_match: float = 2.0
    acoustic_match: float = 1.0
    acoustic_threshold: float = 0.5


# The default weighting, matching the original hard-coded values. Safe to use as
# a default argument precisely BECAUSE ScoringWeights is frozen: the classic
# Python trap is a mutable default like `def f(items=[])`, where every call
# shares one list and changes leak between calls. A frozen dataclass cannot be
# modified, so every caller sees the same unchanging values.
DEFAULT_WEIGHTS = ScoringWeights()


def _to_number(raw: str, field: str, whole: bool = False) -> float:
    """Convert one CSV text value to a number, or explain why it cannot be.

    `float("loud")` raises a bare `ValueError: could not convert string to float:
    'loud'`, which does not say WHICH column was wrong. Wrapping it lets us name
    the field, which is the difference between an error you can act on and one
    you have to go hunting for.
    """
    try:
        return int(raw) if whole else float(raw)
    except (TypeError, ValueError):
        kind = "whole number" if whole else "number"
        raise InvalidSongError(f"{field} must be a {kind}, got {raw!r}") from None


def _song_from_row(row: Dict[str, str]) -> Song:
    """Build one Song from one CSV row, converting text to numbers.

    Range and emptiness checks are NOT repeated here — `Song.__post_init__` does
    them. This function only handles what is specific to reading a CSV: turning
    strings into numbers. Each rule lives in exactly one place.
    """
    return Song(
        id=int(_to_number(row["id"], "id", whole=True)),
        title=(row["title"] or "").strip(),
        artist=(row["artist"] or "").strip(),
        genre=(row["genre"] or "").strip(),
        mood=(row["mood"] or "").strip(),
        energy=_to_number(row["energy"], "energy"),
        tempo_bpm=_to_number(row["tempo_bpm"], "tempo_bpm"),
        valence=_to_number(row["valence"], "valence"),
        danceability=_to_number(row["danceability"], "danceability"),
        acousticness=_to_number(row["acousticness"], "acousticness"),
    )


def _read_rows(csv_path: str) -> Iterator[Tuple[int, Dict[str, str]]]:
    """Yield (file line number, row) pairs, after checking the header.

    Line numbers start at 2 because line 1 is the header. Reporting them lets an
    error message point at the exact line to open in a spreadsheet.
    """
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            missing = [column for column in REQUIRED_COLUMNS if column not in header]
            if missing:
                raise CatalogError(
                    f"{csv_path}: missing required column(s): {', '.join(missing)}. "
                    f"Found: {', '.join(header) if header else '(no header row)'}"
                )
            for line_number, row in enumerate(reader, start=2):
                yield line_number, row
    except FileNotFoundError:
        raise CatalogError(f"catalog file not found: {csv_path}") from None


def load_songs(csv_path: str) -> List[Song]:
    """Read and validate the song catalog, returning a list of Song objects.

    Bad data is never silently accepted. Every problem found is collected and
    reported together in a single CatalogError, rather than raising on the first
    one — so you fix your CSV in one pass instead of re-running once per typo.

    Raises:
        CatalogError: the file is missing, lacks required columns, is empty, has
            duplicate IDs, or contains one or more invalid rows.
    """
    songs: List[Song] = []
    problems: List[str] = []

    for line_number, row in _read_rows(csv_path):
        try:
            songs.append(_song_from_row(row))
        except InvalidSongError as error:
            problems.append(f"line {line_number}: {error}")

    problems.extend(_duplicate_id_problems(songs))

    if problems:
        summary = "\n  ".join(problems)
        raise CatalogError(
            f"{csv_path}: found {len(problems)} problem(s):\n  {summary}"
        )

    if not songs:
        raise CatalogError(f"{csv_path}: catalog is empty (header only, no song rows)")

    return songs


def _duplicate_id_problems(songs: List[Song]) -> List[str]:
    """Report any ID used by more than one song.

    A single Song cannot detect this — it only sees itself. Uniqueness is a
    property of the collection, which is why this check lives in the loader
    rather than in `Song.__post_init__`.
    """
    seen: Dict[int, str] = {}
    problems: List[str] = []
    for song in songs:
        if song.id in seen:
            problems.append(
                f"duplicate id {song.id}: {song.title!r} reuses the id of {seen[song.id]!r}"
            )
        else:
            seen[song.id] = song.title
    return problems


def score_song(
    user: UserProfile, song: Song, weights: ScoringWeights = DEFAULT_WEIGHTS
) -> Tuple[float, List[str]]:
    """Score one song against one listener profile.

    Returns the total score and a list of human-readable reasons. Every reason
    corresponds to points actually awarded, so an explanation can never claim
    something the scorer did not do — including the amount, which is read from
    `weights` rather than written into the text.

    The recipe uses three different kinds of comparison:

        mood, genre   exact match, all-or-nothing
        energy        graded by closeness to the target, on a sliding scale
        acoustic      a continuous value thresholded into yes/no, then compared

    `weights` defaults to DEFAULT_WEIGHTS, so callers that do not care about
    tuning can ignore it entirely. With the defaults the maximum score is 8.0
    (3.0 + 2.0 + 2.0 + 1.0), reachable only when the listener states all four
    preferences and the song matches all of them.
    """
    score = 0.0
    reasons: List[str] = []

    # Categorical matches: an exact string match earns the full weight.
    if user.favorite_mood is not None and song.mood == user.favorite_mood:
        score += weights.mood_match
        reasons.append(f"mood match: {song.mood} (+{weights.mood_match:.1f})")
    if user.favorite_genre is not None and song.genre == user.favorite_genre:
        score += weights.genre_match
        reasons.append(f"genre match: {song.genre} (+{weights.genre_match:.1f})")

    # Numeric closeness: reward songs whose energy is NEAR the target, not just
    # high. Both values sit in 0.0-1.0, so closeness lands in 0.0-1.0 too, and
    # the award is the full energy weight scaled by that closeness.
    if user.target_energy is not None:
        closeness = 1 - abs(song.energy - user.target_energy)
        points = weights.energy_match * closeness
        score += points
        reasons.append(
            f"energy {song.energy} near target {user.target_energy} (+{points:.2f})"
        )

    # Optional acoustic preference: rewards agreement, either direction.
    if user.likes_acoustic is not None:
        is_acoustic = song.acousticness > weights.acoustic_threshold
        if is_acoustic == user.likes_acoustic:
            score += weights.acoustic_match
            reasons.append(
                f"acoustic preference match (+{weights.acoustic_match:.1f})"
            )

    return score, reasons


# How many decimal places to keep when deciding whether two numbers are "equal"
# for ranking purposes. Floating-point noise from arithmetic like
# 2.0 * (1 - abs(0.82 - 0.9)) lands around the 16th decimal place, so rounding at
# 6 erases the noise while preserving any difference a listener could care about
# (scores are displayed to 2 decimals).
TIE_BREAK_PRECISION = 6


def _ranking_key(song: Song, score: float, user: UserProfile) -> Tuple:
    """Build the sort key that decides a song's position, ties included.

    Python sorts tuples element by element: it compares the first items, and only
    looks at the second if the first are equal. That makes a tuple a natural way
    to express "rank by this, break ties with that, then that".

    Every element here is written so that SMALLER WINS, which lets us sort in
    plain ascending order. Values where bigger is better are negated. Mixing
    directions is exactly what `reverse=True` cannot express, since it would flip
    the alphabetical rule too.

    The ladder, in order:

        1. score            highest first          (negated)
        2. energy gap       closest to target first
        3. danceability     highest first          (negated)
        4. title            alphabetical, case-insensitive
        5. id               guarantees a total order; ids are unique

    Rounding matters. Without it, 1.84 and 1.8399999999999999 count as different
    scores and rule 1 settles the contest on floating-point noise — an ordering
    no user could ever be given a reason for. Rounding makes them genuinely tied
    so a rule with a real explanation gets its turn.

    Rule 5 exists so the result is never ambiguous. Two songs could in principle
    share a title; ids cannot collide, because the loader rejects duplicates.
    """
    energy_gap = (
        abs(song.energy - user.target_energy) if user.target_energy is not None else 0.0
    )
    return (
        -round(score, TIE_BREAK_PRECISION),
        round(energy_gap, TIE_BREAK_PRECISION),
        -round(song.danceability, TIE_BREAK_PRECISION),
        song.title.lower(),
        song.id,
    )


def recommend_songs(
    user: UserProfile,
    songs: List[Song],
    k: int = 5,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
) -> List[Tuple[Song, float, str]]:
    """Score every song, then return the top k as (song, score, explanation).

    This is the one and only ranking implementation. Scoring grades a single
    song; ranking is what turns those grades into a recommendation.

    Ordering is fully deterministic: `_ranking_key` spells out an explicit
    tie-breaker ladder, so the result never depends on catalog order or on
    floating-point noise. See that function for the rules.

    `.sort()` runs on a new list, so the caller's `songs` list is never
    reordered.
    """
    scored: List[Tuple[Song, float, str]] = []
    for song in songs:
        score, reasons = score_song(user, song, weights)
        explanation = "; ".join(reasons) if reasons else "no strong matches"
        scored.append((song, score, explanation))

    # Ascending, with no reverse=True: the key already encodes every direction.
    scored.sort(key=lambda item: _ranking_key(item[0], item[1], user))
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

    def __init__(self, songs: List[Song], weights: ScoringWeights = DEFAULT_WEIGHTS):
        self.songs = songs
        self.weights = weights

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top k Songs for this user, ranked highest score first."""
        ranked = recommend_songs(user, self.songs, k, self.weights)
        return [song for song, _score, _explanation in ranked]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable string explaining a song's score."""
        score, reasons = score_song(user, song, self.weights)
        detail = "; ".join(reasons) if reasons else "no strong matches"
        return f"Score {score:.2f} — {detail}"
