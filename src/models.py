"""Data models for VibeFlow.

These are the shapes the rest of the system passes around: what a song is, what
a listener's stated taste is, and how much each feature counts.

They live in their own module because they have no dependencies on scoring,
ranking, or file loading — but almost everything else depends on them. Keeping
them separate means a future journey planner or sequencer can import `Song`
without dragging in the CSV loader.

The models validate themselves. A Song that breaks its own rules cannot be
constructed at all, from any source.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.exceptions import InvalidSongError


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
