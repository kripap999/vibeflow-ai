"""Tests for the music recommender.

Two kinds of test live here.

1. The two original starter tests, which exercise the Recommender class.

2. CHARACTERIZATION tests, added before the dict/object consolidation. A
   characterization test records what the code does *today*, not what it ideally
   should do. Its job is to go red if a refactor accidentally changes behavior.
   Think of it as a net under a trapeze: it does not make the trick better, it
   makes falling safe.

The numbers asserted below (6.84, 8.0, 3.94, 7.84, 2.0, 0.36) were captured from
the dict-based implementation BEFORE the refactor and deliberately left
unchanged afterwards. Only the way inputs are *constructed* changed. That is the
proof the consolidation preserved behavior: same numbers, one code path.

A few tests are marked KNOWN DEFECT. Those pin behavior we have already agreed
is wrong. We keep them so the bug stays visible, and so that when we fix it the
test fails *on purpose* — telling us the change landed where we intended.
"""

from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from src.exceptions import CatalogError, InvalidSongError, VibeFlowError
from src.recommender import (
    DEFAULT_WEIGHTS,
    REQUIRED_COLUMNS,
    Recommender,
    ScoringWeights,
    Song,
    UserProfile,
    load_songs,
    recommend_songs,
    score_song,
)

CSV_HEADER = ",".join(REQUIRED_COLUMNS)
VALID_ROW = "1,Sunrise City,Neon Echo,pop,happy,0.82,118,0.84,0.79,0.18"
SECOND_VALID_ROW = "2,Library Rain,Paper Lanterns,lofi,chill,0.35,72,0.60,0.58,0.86"


def write_csv(tmp_path, *rows: str, header: str = CSV_HEADER) -> str:
    """Write a throwaway CSV for failure tests and return its path.

    `tmp_path` is a pytest fixture giving each test its own empty directory,
    cleaned up automatically. It lets us test bad data without ever touching the
    real data/songs.csv.
    """
    path = tmp_path / "songs.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return str(path)

# Build the CSV path from this file's location rather than hard-coding
# "data/songs.csv". That relative string only works if pytest happens to be run
# from the repository root; this works no matter where you run it from.
REPO_ROOT = Path(__file__).resolve().parent.parent
SONGS_CSV = str(REPO_ROOT / "data" / "songs.csv")

# These mirror src/main.py, so the numbers asserted here are the same numbers
# printed by the real application. Both deliberately leave some fields unset.
POP_PROFILE = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.9)
LOFI_PROFILE = UserProfile(
    favorite_genre="lofi",
    favorite_mood="chill",
    target_energy=0.35,
    likes_acoustic=True,
)

EXPECTED_FIELDS = {
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
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1,
            title="Test Pop Track",
            artist="Test Artist",
            genre="pop",
            mood="happy",
            energy=0.8,
            tempo_bpm=120,
            valence=0.9,
            danceability=0.8,
            acousticness=0.2,
        ),
        Song(
            id=2,
            title="Chill Lofi Loop",
            artist="Test Artist",
            genre="lofi",
            mood="chill",
            energy=0.4,
            tempo_bpm=80,
            valence=0.6,
            danceability=0.5,
            acousticness=0.9,
        ),
    ]
    return Recommender(songs)


@pytest.fixture
def catalog() -> list:
    """The real 19-song catalog, loaded once per test that asks for it."""
    return load_songs(SONGS_CSV)


@pytest.fixture
def by_title(catalog) -> dict:
    """The catalog keyed by song title, for readable lookups in assertions."""
    return {song.title: song for song in catalog}


# --------------------------------------------------------------------------
# Original starter tests
# --------------------------------------------------------------------------

def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    # Starter expectation: the pop, happy, high energy song should score higher
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# --------------------------------------------------------------------------
# load_songs: reading the CSV
# --------------------------------------------------------------------------

def test_load_songs_reads_every_row(catalog):
    """The loader should find all 19 songs in the CSV (20 lines minus header)."""
    assert len(catalog) == 19


def test_load_songs_returns_song_objects(catalog):
    """The loader now produces the canonical Song model, not plain dicts.

    Before the refactor this test asserted `isinstance(row, dict)`. Flipping it
    is the whole point of the consolidation: the application and the tests now
    share one representation, so the `Song` dataclass is no longer reachable
    only from test code.
    """
    assert all(isinstance(song, Song) for song in catalog)


def test_song_model_declares_every_expected_field():
    """The dataclass itself now guarantees the schema.

    Previously a test looped over all 19 rows checking each dict had the right
    keys, because a dict can be missing anything. A Song cannot: omitting a
    field raises TypeError at construction. So the check moves from "inspect the
    data" to "inspect the model" — one assertion instead of nineteen.
    """
    assert {f.name for f in fields(Song)} == EXPECTED_FIELDS


def test_load_songs_converts_text_to_numbers(by_title):
    """CSV files store everything as text; the loader must convert to numbers.

    Without this, "0.82" stays a string and `abs(song.energy - target)` would
    fail, because you cannot subtract a number from text. Note tempo_bpm becomes
    a float (118.0), not an int.
    """
    song = by_title["Sunrise City"]

    assert isinstance(song.id, int)
    assert song.id == 1

    for field_name in ("energy", "tempo_bpm", "valence", "danceability", "acousticness"):
        assert isinstance(getattr(song, field_name), float), f"{field_name} should be a float"

    assert song.energy == pytest.approx(0.82)
    assert song.tempo_bpm == pytest.approx(118.0)

    # Text fields stay text.
    assert song.title == "Sunrise City"
    assert song.artist == "Neon Echo"


def test_load_songs_preserves_csv_order(catalog):
    """Rows come back in file order.

    This used to matter for ranking, because Python's stable sort meant CSV
    order silently decided ties. It no longer does — `_ranking_key` spells the
    tie-breakers out explicitly — but the loader should still hand back the file
    as written, so a reader can line results up against the spreadsheet.
    """
    assert catalog[0].title == "Sunrise City"
    assert catalog[-1].title == "Dust Road Home"


def test_catalog_ids_are_unique(catalog):
    """A data guard, not a characterization test.

    Nothing in the code enforces unique IDs yet. This asserts the data file is
    currently clean, so if someone pastes in a duplicate ID while expanding the
    catalog to 75+ tracks, we hear about it immediately.
    """
    ids = [song.id for song in catalog]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------
# score_song: the scoring recipe
# --------------------------------------------------------------------------

def test_score_song_awards_maximum_eight_points(by_title):
    """A perfect match on all four features scores exactly 8.0.

    3.0 (mood) + 2.0 (genre) + 2.0 (energy dead-on) + 1.0 (acoustic) = 8.0.
    'Library Rain' has energy 0.35, identical to the profile's target, so the
    energy term pays out in full. This is the highest score the recipe can give.
    """
    score, _ = score_song(LOFI_PROFILE, by_title["Library Rain"])
    assert score == pytest.approx(8.0)


def test_score_song_matches_documented_pop_score(by_title):
    """Pins the exact number the README and model card report: 6.84.

    3.0 (mood happy) + 2.0 (genre pop) + 1.84 (energy 0.82 vs target 0.9).
    The energy term is 2.0 * (1 - |0.82 - 0.9|) = 2.0 * 0.92 = 1.84.
    If a refactor changes any weight, this breaks.
    """
    score, _ = score_song(POP_PROFILE, by_title["Sunrise City"])
    assert score == pytest.approx(6.84)


def test_score_song_partial_match_scores_lower(by_title):
    """'Gym Hero' matches genre and energy but NOT mood, so it scores 3.94.

    This is the case the model card calls out: a pop song with the wrong mood
    still ranks well because two of three signals agree. Worth pinning, because
    it is behavior we may deliberately change later.
    """
    score, _ = score_song(POP_PROFILE, by_title["Gym Hero"])
    assert score == pytest.approx(3.94)


def test_score_song_returns_one_reason_per_awarded_component(by_title):
    """The explanation must be built from real awarded points, not invented text.

    This is the seed of VibeFlow's 'grounded explanation' rule: every sentence
    shown to a user traces back to a specific point the scorer actually gave.
    """
    _, reasons = score_song(POP_PROFILE, by_title["Sunrise City"])

    assert len(reasons) == 3
    assert "mood match: happy (+3.0)" in reasons
    assert "genre match: pop (+2.0)" in reasons
    assert any("energy 0.82 near target 0.9" in r for r in reasons)


def test_score_song_with_no_stated_preferences_scores_zero(by_title):
    """An empty profile means no points and no reasons.

    Every scoring rule is guarded by `is not None`, so a profile that states
    nothing skips all of them rather than crashing.
    """
    score, reasons = score_song(UserProfile(), by_title["Sunrise City"])

    assert score == 0.0
    assert reasons == []


def test_score_song_skips_unstated_preferences(by_title):
    """Stating genre alone must not require the other fields to be set."""
    score, reasons = score_song(UserProfile(favorite_genre="pop"), by_title["Sunrise City"])

    assert score == pytest.approx(2.0)
    assert len(reasons) == 1


def test_unstated_preference_is_not_the_same_as_stating_false(by_title):
    """The distinction that makes this refactor behavior-preserving.

    'Sunrise City' is not acoustic (acousticness 0.18). A listener who says
    nothing about acoustics gets no acoustic points. A listener who explicitly
    says likes_acoustic=False *agrees* with the song and earns +1.0.

    Collapsing None into False would have raised every score in the application
    by 1.0 and invalidated the numbers in the README and model card.
    """
    song = by_title["Sunrise City"]

    unstated, _ = score_song(POP_PROFILE, song)
    stated_false, _ = score_song(
        UserProfile(
            favorite_genre="pop",
            favorite_mood="happy",
            target_energy=0.9,
            likes_acoustic=False,
        ),
        song,
    )

    assert unstated == pytest.approx(6.84)
    assert stated_false == pytest.approx(7.84)


def test_score_song_energy_reward_is_graded_not_all_or_nothing(by_title):
    """Energy is scored by CLOSENESS, so it pays out on a sliding scale.

    Exact match     -> the full 2.0
    Far from target -> a small fraction

    This is the most important idea in the current scorer and the one that
    carries straight into VibeFlow's journey planning: we reward songs NEAR a
    target energy, not simply the loudest songs available.
    """
    song = by_title["Sunrise City"]  # energy 0.82

    exact, _ = score_song(UserProfile(target_energy=0.82), song)
    far, _ = score_song(UserProfile(target_energy=0.0), song)

    assert exact == pytest.approx(2.0)
    assert far == pytest.approx(0.36)  # 2.0 * (1 - 0.82)
    assert exact > far


def test_score_song_acoustic_preference_rewards_agreement_both_ways(by_title):
    """+1.0 is awarded when the song AGREES with the preference, either way.

    A song counts as acoustic when acousticness > 0.5. So a listener who says
    likes_acoustic=False earns the point on every NON-acoustic song. This is
    symmetric agreement, not a bonus for acoustic songs.
    """
    electronic = by_title["Sunrise City"]  # acousticness 0.18 -> not acoustic
    acoustic = by_title["Library Rain"]  # acousticness 0.86 -> acoustic

    assert score_song(UserProfile(likes_acoustic=False), electronic)[0] == pytest.approx(1.0)
    assert score_song(UserProfile(likes_acoustic=True), electronic)[0] == pytest.approx(0.0)
    assert score_song(UserProfile(likes_acoustic=True), acoustic)[0] == pytest.approx(1.0)
    assert score_song(UserProfile(likes_acoustic=False), acoustic)[0] == pytest.approx(0.0)


def test_out_of_range_energy_is_now_rejected_FORMERLY_KNOWN_DEFECT():
    """This replaces a KNOWN_DEFECT test. The defect is fixed.

    Previously Song(energy=5.0) was constructible, and scoring it returned -8.0
    with the malformed explanation '(+-8.00)' — no exception, no warning. A
    single CSV typo could corrupt every ranking silently.

    Now the bad state cannot exist: Song validates its own invariants in
    __post_init__, so there is no way to reach the scorer with energy=5.0. The
    old test asserted the broken behavior; this one asserts the guard.
    """
    with pytest.raises(InvalidSongError, match="energy must be between 0.0 and 1.0"):
        Song(
            id=999,
            title="Out Of Range",
            artist="Bad Data",
            genre="pop",
            mood="happy",
            energy=5.0,
            tempo_bpm=120.0,
            valence=0.5,
            danceability=0.5,
            acousticness=0.5,
        )


# --------------------------------------------------------------------------
# recommend_songs: ranking and top-k
# --------------------------------------------------------------------------

def test_recommend_songs_returns_exactly_k_results(catalog):
    assert len(recommend_songs(POP_PROFILE, catalog, k=5)) == 5
    assert len(recommend_songs(POP_PROFILE, catalog, k=1)) == 1


def test_recommend_songs_returns_song_score_explanation_triples(catalog):
    """Documents the return SHAPE: a 3-tuple of (Song, float, str).

    src/main.py unpacks exactly this shape, so anything that changes it breaks
    the application.
    """
    results = recommend_songs(POP_PROFILE, catalog, k=3)

    for song, score, explanation in results:
        assert isinstance(song, Song)
        assert isinstance(score, float)
        assert isinstance(explanation, str)
        assert explanation.strip() != ""


def test_recommend_songs_sorts_highest_score_first(catalog):
    """The ranking rule: scores must descend down the list.

    Scoring grades one song at a time; ranking is what turns those grades into
    a recommendation. This test checks the ranking half.
    """
    scores = [score for _, score, _ in recommend_songs(POP_PROFILE, catalog, k=19)]

    assert scores == sorted(scores, reverse=True)


def test_recommend_songs_top_pick_for_pop_profile(catalog):
    """Pins the actual top 3 the application prints today."""
    titles = [song.title for song, _, _ in recommend_songs(POP_PROFILE, catalog, k=3)]

    assert titles == ["Sunrise City", "Rooftop Lights", "Gym Hero"]


def test_recommend_songs_does_not_modify_the_catalog(catalog):
    """Ranking must not reorder or damage the caller's list.

    recommend_songs builds a new list and sorts that, leaving the input alone. A
    function that quietly rearranges its argument causes bugs that are very hard
    to trace, so this is worth locking down.
    """
    before = [song.title for song in catalog]

    recommend_songs(POP_PROFILE, catalog, k=5)

    assert [song.title for song in catalog] == before


def test_recommend_songs_explanation_matches_score_reasons(catalog, by_title):
    """The displayed explanation is the scorer's reasons joined by '; '.

    Confirms nothing is added or dropped between scoring and display.
    """
    results = recommend_songs(POP_PROFILE, catalog, k=1)
    song, _, explanation = results[0]

    _, reasons = score_song(POP_PROFILE, by_title[song.title])
    assert explanation == "; ".join(reasons)


def test_float_noise_no_longer_decides_ties_FORMERLY_KNOWN_DEFECT(catalog):
    """This replaces a KNOWN_DEFECT test. The defect is fixed.

    For the rock profile two songs both display 1.84, but their raw floats differ:

        Sunrise City -> 1.8399999999999999
        Iron Verdict -> 1.84

    Both come from 2.0 * (1 - 0.08); |0.82-0.9| and |0.98-0.9| simply land on
    different binary floats. Previously Iron Verdict won on that 16th-decimal
    difference — an ordering no user could be given a reason for.

    Now rounding makes them genuinely tied on score AND on energy gap, so the
    ladder falls through to danceability, where Sunrise City (0.79) beats Iron
    Verdict (0.45). The order flipped, and the new order has an explanation.
    """
    rock_profile = UserProfile(
        favorite_genre="rock", favorite_mood="intense", target_energy=0.9
    )
    ranked = recommend_songs(rock_profile, catalog, k=19)
    titles = [song.title for song, _, _ in ranked]

    assert titles.index("Sunrise City") < titles.index("Iron Verdict")


# --------------------------------------------------------------------------
# One code path: the Recommender is now a thin facade
# --------------------------------------------------------------------------

def test_score_song_consumes_song_objects_directly(by_title):
    """THE POINT OF THIS REFACTOR.

    Before, this test fed the same catalog row through two paths — score_song on
    a dict, and Recommender._score on a Song — and asserted they agreed at 7.84.

    That adapter is now deleted. score_song takes a Song directly, so there is
    nothing left to disagree. The number is unchanged, which is the evidence
    that removing the duplication did not change results.
    """
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.9,
        likes_acoustic=False,
    )

    score, reasons = score_song(user, by_title["Sunrise City"])

    assert score == pytest.approx(7.84)
    assert len(reasons) == 4


def test_recommender_delegates_to_recommend_songs(catalog):
    """Recommender.recommend must be a pass-through, not a second ranker.

    It previously ran its own `sorted(...)` call. Now it forwards to
    recommend_songs and drops the scores. Asserting both produce the same order
    over all 19 real songs is what proves the delegation is faithful.
    """
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.9,
        likes_acoustic=False,
    )

    function_titles = [song.title for song, _, _ in recommend_songs(user, catalog, k=5)]
    class_titles = [song.title for song in Recommender(catalog).recommend(user, k=5)]

    assert function_titles == class_titles


def test_explain_recommendation_adds_a_score_prefix(by_title):
    """Documents the one remaining difference between the two entry points.

    recommend_songs returns only the joined reasons:
        "mood match: happy (+3.0); genre match: pop (+2.0); ..."

    Recommender.explain_recommendation prepends the total:
        "Score 7.84 — mood match: happy (+3.0); ..."

    Both read the same reasons from the same scorer, so the facts cannot drift.
    Only the presentation differs.
    """
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.9,
        likes_acoustic=False,
    )
    song = by_title["Sunrise City"]

    explanation = Recommender([song]).explain_recommendation(user, song)

    assert explanation.startswith("Score 7.84 — ")
    assert "mood match: happy (+3.0)" in explanation


# --------------------------------------------------------------------------
# ScoringWeights: the weighting is now configuration, not code
# --------------------------------------------------------------------------

def test_default_weights_match_the_original_hard_coded_values():
    """The defaults must reproduce the numbers that used to be literals.

    This is what makes introducing ScoringWeights behavior-preserving. If anyone
    changes a default, every documented score in README.md and model_card.md
    becomes wrong — so the defaults are pinned here explicitly.
    """
    assert DEFAULT_WEIGHTS.mood_match == pytest.approx(3.0)
    assert DEFAULT_WEIGHTS.genre_match == pytest.approx(2.0)
    assert DEFAULT_WEIGHTS.energy_match == pytest.approx(2.0)
    assert DEFAULT_WEIGHTS.acoustic_match == pytest.approx(1.0)
    assert DEFAULT_WEIGHTS.acoustic_threshold == pytest.approx(0.5)


def test_scoring_weights_cannot_be_modified():
    """ScoringWeights is frozen, so a weighting cannot change mid-run.

    This is also what makes DEFAULT_WEIGHTS safe as a default argument. The
    classic Python trap is a mutable default like `def f(items=[])`: every call
    shares one object, so a change made by one caller leaks into the next.
    A frozen dataclass raises instead of mutating.
    """
    weights = ScoringWeights()

    with pytest.raises(FrozenInstanceError):
        weights.mood_match = 99.0


def test_custom_weights_change_the_score(by_title):
    """Passing different weights produces a different total.

    Doubling the mood weight from 3.0 to 6.0 should add exactly 3.0 to a song
    that matches on mood: 6.84 becomes 9.84.
    """
    song = by_title["Sunrise City"]

    default_score, _ = score_song(POP_PROFILE, song)
    louder_mood, _ = score_song(POP_PROFILE, song, ScoringWeights(mood_match=6.0))

    assert default_score == pytest.approx(6.84)
    assert louder_mood == pytest.approx(9.84)


def test_zero_weights_score_nothing(by_title):
    """A weighting of all zeros scores every song at 0.0.

    Useful as a sanity check that no points are awarded from anywhere except the
    weights — no stray literal is hiding in the scorer.
    """
    silent = ScoringWeights(
        mood_match=0.0, genre_match=0.0, energy_match=0.0, acoustic_match=0.0
    )

    score, reasons = score_song(LOFI_PROFILE, by_title["Library Rain"], silent)

    assert score == pytest.approx(0.0)
    # Reasons are still produced — the rules matched, they were just worth nothing.
    assert len(reasons) == 4


def test_explanation_reports_the_weight_actually_used(by_title):
    """Grounded explanations: the text must quote the real weight, not '3.0'.

    The reason strings used to hard-code '(+3.0)' as literal text. If a weight
    changed, the explanation would have lied about how many points were awarded.
    Now the number is read from the weights, so text and arithmetic cannot drift.
    """
    song = by_title["Sunrise City"]

    _, reasons = score_song(POP_PROFILE, song, ScoringWeights(mood_match=5.0))

    assert "mood match: happy (+5.0)" in reasons
    assert "mood match: happy (+3.0)" not in reasons


def test_acoustic_threshold_is_configurable(by_title):
    """The '> 0.5' acousticness cutoff is a tuning knob, not a law of nature.

    'Sunrise City' has acousticness 0.18. Under the default 0.5 threshold it is
    not acoustic. Drop the threshold to 0.1 and it counts as acoustic, which
    flips which preference it agrees with.
    """
    song = by_title["Sunrise City"]
    wants_acoustic = UserProfile(likes_acoustic=True)

    assert score_song(wants_acoustic, song)[0] == pytest.approx(0.0)

    lenient = ScoringWeights(acoustic_threshold=0.1)
    assert score_song(wants_acoustic, song, lenient)[0] == pytest.approx(1.0)


def test_weights_flow_through_the_recommender_class(catalog):
    """The facade must honour its configured weights, not silently use defaults."""
    loud_mood = ScoringWeights(mood_match=6.0)
    rec = Recommender(catalog, weights=loud_mood)

    explanation = rec.explain_recommendation(POP_PROFILE, catalog[0])

    assert "mood match: happy (+6.0)" in explanation
    assert explanation.startswith("Score 9.84 — ")


def test_model_card_weight_experiment_changes_scores_but_not_order(catalog):
    """Reproduces the weight experiment documented in model_card.md section 7.

    The model card reports setting energy to 4.0 and genre to 1.0, and concludes:
    the #1 pick stayed the same for every profile, but "mid-list rankings
    (positions 3-5) became energy-driven and genre matches barely mattered."

    The first half is correct. The second half is NOT: with this catalog the
    experiment changes no positions at all — not the top 5, not any of the 19.
    Scores move a lot (Sunrise City 6.84 -> 7.68) while the ORDER is identical,
    because doubling the energy weight scales every song's energy term by the
    same factor, and the score gaps are wide enough that halving the genre bonus
    never flips an adjacent pair.

    This test exists to keep the documented claim honest. The model card should
    be corrected to say the ranking was unchanged.
    """
    experiment = ScoringWeights(energy_match=4.0, genre_match=1.0)

    default_order = [s.title for s, _, _ in recommend_songs(POP_PROFILE, catalog, 19)]
    experiment_order = [
        s.title for s, _, _ in recommend_songs(POP_PROFILE, catalog, 19, experiment)
    ]
    assert default_order == experiment_order

    default_top, _ = score_song(POP_PROFILE, catalog[0])
    experiment_top, _ = score_song(POP_PROFILE, catalog[0], experiment)
    assert default_top == pytest.approx(6.84)
    assert experiment_top == pytest.approx(7.68)


# --------------------------------------------------------------------------
# Song validation: a Song cannot exist in an invalid state
# --------------------------------------------------------------------------

def valid_song_kwargs(**overrides):
    """A complete set of valid Song arguments, with selected fields replaced.

    Lets each test change exactly one field to something bad, so the failure it
    triggers is unambiguous.
    """
    kwargs = dict(
        id=1,
        title="Test Track",
        artist="Test Artist",
        genre="pop",
        mood="happy",
        energy=0.5,
        tempo_bpm=120.0,
        valence=0.5,
        danceability=0.5,
        acousticness=0.5,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize("field", ["energy", "valence", "danceability", "acousticness"])
def test_song_rejects_proportions_above_one(field):
    """All four 0.0-1.0 fields are range-checked, not just energy.

    Parametrize runs this once per field and reports each as its own test, so a
    failure names the offending field instead of hiding inside a loop.
    """
    with pytest.raises(InvalidSongError, match=f"{field} must be between 0.0 and 1.0"):
        Song(**valid_song_kwargs(**{field: 1.5}))


@pytest.mark.parametrize("field", ["energy", "valence", "danceability", "acousticness"])
def test_song_rejects_negative_proportions(field):
    with pytest.raises(InvalidSongError, match=f"{field} must be between 0.0 and 1.0"):
        Song(**valid_song_kwargs(**{field: -0.1}))


@pytest.mark.parametrize("boundary", [0.0, 1.0])
def test_song_accepts_the_range_boundaries(boundary):
    """0.0 and 1.0 are valid. The check is inclusive on both ends.

    Boundary values are where off-by-one errors live, so they get their own test.
    'Spacewalk Thoughts' style ambient tracks sit near 0, and 'Iron Verdict'
    style tracks sit near 1, so both ends are real data, not hypothetical.
    """
    song = Song(**valid_song_kwargs(energy=boundary, acousticness=boundary))
    assert song.energy == pytest.approx(boundary)


@pytest.mark.parametrize("field", ["title", "artist", "genre", "mood"])
@pytest.mark.parametrize("bad_value", ["", "   "])
def test_song_rejects_blank_text_fields(field, bad_value):
    """Empty and whitespace-only strings are both rejected.

    A song titled "   " is as useless as one titled "", but a plain `if not
    value` check would accept the spaces. `.strip()` is what catches it.
    """
    with pytest.raises(InvalidSongError, match=f"{field} must be a non-empty string"):
        Song(**valid_song_kwargs(**{field: bad_value}))


@pytest.mark.parametrize("bad_tempo", [0, -120.0])
def test_song_rejects_non_positive_tempo(bad_tempo):
    """A song at 0 BPM is not a song. Tempo must be strictly greater than 0."""
    with pytest.raises(InvalidSongError, match="tempo_bpm must be greater than 0"):
        Song(**valid_song_kwargs(tempo_bpm=bad_tempo))


def test_song_rejects_non_numeric_values():
    """Type hints are documentation, not enforcement.

    `energy: float` does not stop Python assigning the string "loud". Only an
    explicit runtime check does, which is the whole reason __post_init__ exists.
    """
    with pytest.raises(InvalidSongError, match="energy must be a number"):
        Song(**valid_song_kwargs(energy="loud"))


def test_song_rejects_booleans_where_numbers_belong():
    """In Python, `True` IS `1` and `isinstance(True, int)` is True.

    So a naive numeric check would silently accept energy=True as energy=1.0.
    The validator explicitly excludes bool to close that hole.
    """
    with pytest.raises(InvalidSongError, match="energy must be a number"):
        Song(**valid_song_kwargs(energy=True))


def test_song_reports_every_problem_at_once():
    """One error message lists all the problems, not just the first.

    Fixing bad data one error per run is miserable. Aggregating means you see the
    whole picture in a single pass.
    """
    with pytest.raises(InvalidSongError) as caught:
        Song(**valid_song_kwargs(title="", energy=9.0, tempo_bpm=-5))

    message = str(caught.value)
    assert "title must be a non-empty string" in message
    assert "energy must be between 0.0 and 1.0" in message
    assert "tempo_bpm must be greater than 0" in message


def test_user_profile_rejects_target_energy_outside_range():
    """Both sides of the energy comparison must be in range.

    Validating songs alone would not close the negative-score hole: a target of
    5.0 against a perfectly valid song still gives 2.0 * (1 - 4.5) = -7.0.
    """
    with pytest.raises(InvalidSongError, match="target_energy must be between 0.0 and 1.0"):
        UserProfile(target_energy=5.0)


def test_user_profile_allows_an_unstated_target_energy():
    """None is still valid — it means "the listener did not say"."""
    assert UserProfile().target_energy is None


# --------------------------------------------------------------------------
# Catalog validation: failure modes of load_songs
# --------------------------------------------------------------------------

def test_the_real_catalog_passes_validation(catalog):
    """The shipped data file is valid under all the new rules.

    Worth asserting explicitly: it would be embarrassing to add validation that
    rejects our own data.
    """
    assert len(catalog) == 19


def test_load_songs_reports_a_missing_file():
    """A missing catalog gives a clear message, not a raw FileNotFoundError."""
    with pytest.raises(CatalogError, match="catalog file not found"):
        load_songs("data/does_not_exist.csv")


def test_load_songs_reports_missing_columns(tmp_path):
    """A renamed or dropped column fails at load time, naming what is missing.

    Previously this produced `KeyError: 'energy'` from deep inside the parse
    loop, which does not tell you the CSV header is wrong.
    """
    header = "id,title,artist,genre,mood"  # missing five numeric columns
    path = write_csv(tmp_path, "1,Song,Artist,pop,happy", header=header)

    with pytest.raises(CatalogError, match="missing required column"):
        load_songs(path)


def test_missing_column_error_names_every_absent_column(tmp_path):
    header = "id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability"
    path = write_csv(tmp_path, "1,Song,Artist,pop,happy,0.5,120,0.5,0.5", header=header)

    with pytest.raises(CatalogError) as caught:
        load_songs(path)

    assert "acousticness" in str(caught.value)


def test_load_songs_reports_an_empty_catalog(tmp_path):
    """A header with no rows is an error, not an empty recommendation list.

    Returning [] would make the app print 'Loaded songs: 0' and then silently
    recommend nothing, which looks like a logic bug rather than a data problem.
    """
    path = write_csv(tmp_path)

    with pytest.raises(CatalogError, match="catalog is empty"):
        load_songs(path)


def test_load_songs_reports_duplicate_ids(tmp_path):
    """Uniqueness is a property of the collection, so the loader checks it.

    A single Song cannot detect this — it only ever sees itself.
    """
    duplicate = "1,Different Song,Other Artist,rock,intense,0.9,150,0.4,0.6,0.1"
    path = write_csv(tmp_path, VALID_ROW, duplicate)

    with pytest.raises(CatalogError, match="duplicate id 1"):
        load_songs(path)


def test_load_songs_reports_the_line_number_of_a_bad_row(tmp_path):
    """Errors point at the exact file line, so you can open it and fix it.

    Line 3 here: line 1 is the header, line 2 is the good row.
    """
    bad = "3,Broken,Artist,pop,happy,NOT_A_NUMBER,120,0.5,0.5,0.5"
    path = write_csv(tmp_path, VALID_ROW, bad)

    with pytest.raises(CatalogError) as caught:
        load_songs(path)

    message = str(caught.value)
    assert "line 3" in message
    assert "energy must be a number" in message


def test_load_songs_reports_out_of_range_values_with_context(tmp_path):
    bad = "3,Too Loud,Artist,pop,happy,5.0,120,0.5,0.5,0.5"
    path = write_csv(tmp_path, VALID_ROW, bad)

    with pytest.raises(CatalogError, match="energy must be between 0.0 and 1.0"):
        load_songs(path)


def test_load_songs_aggregates_problems_across_multiple_rows(tmp_path):
    """Every bad row is reported in one error, with a count.

    This is the payoff of collecting instead of failing fast: three typos take
    one run to find, not three.
    """
    path = write_csv(
        tmp_path,
        VALID_ROW,
        "3,Bad Energy,Artist,pop,happy,5.0,120,0.5,0.5,0.5",
        "4,,Artist,pop,happy,0.5,120,0.5,0.5,0.5",
        "5,Bad Tempo,Artist,pop,happy,0.5,0,0.5,0.5,0.5",
    )

    with pytest.raises(CatalogError) as caught:
        load_songs(path)

    message = str(caught.value)
    assert "found 3 problem(s)" in message
    assert "line 3" in message and "line 4" in message and "line 5" in message


def test_load_songs_accepts_extra_columns(tmp_path):
    """Unknown columns are ignored rather than rejected.

    This lets the data file carry notes or future fields without breaking the
    loader — useful when the schema grows in Module 2.
    """
    header = CSV_HEADER + ",notes"
    path = write_csv(tmp_path, VALID_ROW + ",added for testing", header=header)

    songs = load_songs(path)

    assert len(songs) == 1
    assert songs[0].title == "Sunrise City"


def test_load_songs_strips_surrounding_whitespace(tmp_path):
    """Stray spaces in a hand-edited CSV should not become part of a title."""
    padded = "1,  Sunrise City  ,  Neon Echo  ,pop,happy,0.82,118,0.84,0.79,0.18"
    path = write_csv(tmp_path, padded)

    song = load_songs(path)[0]

    assert song.title == "Sunrise City"
    assert song.artist == "Neon Echo"


def test_catalog_errors_share_a_common_base():
    """Both error types inherit VibeFlowError, so one handler can catch either.

    This is what lets a future Streamlit app show a friendly message for any
    expected failure while still letting real programming bugs crash loudly.
    """
    assert issubclass(CatalogError, VibeFlowError)
    assert issubclass(InvalidSongError, VibeFlowError)

    with pytest.raises(VibeFlowError):
        load_songs("data/does_not_exist.csv")


# --------------------------------------------------------------------------
# Deterministic tie-breaking: every rung of the ladder
# --------------------------------------------------------------------------

# A weighting that isolates one rule at a time. Turning mood and acoustic off
# makes it easy to build songs that tie on score for a known reason.
TIE_WEIGHTS = ScoringWeights(
    mood_match=0.0, genre_match=1.0, energy_match=2.0, acoustic_match=0.0
)


def tie_song(song_id: int, title: str, **overrides) -> Song:
    """A song with neutral values, so each test varies only what it cares about."""
    return Song(**valid_song_kwargs(id=song_id, title=title, **overrides))


def ranked_titles(user, songs, weights=TIE_WEIGHTS):
    return [song.title for song, _, _ in recommend_songs(user, songs, len(songs), weights)]


def test_rung_one_higher_score_always_wins():
    """Score dominates. No later rule can promote a lower-scoring song.

    'Loser' has better danceability and an alphabetically earlier title, which
    would win rules 3 and 4 — but it never gets that far, because rule 1 settles
    it first.
    """
    user = UserProfile(favorite_genre="pop", target_energy=0.5)
    winner = tie_song(1, "Zebra Winner", genre="pop", energy=0.5, danceability=0.1)
    loser = tie_song(2, "Apple Loser", genre="rock", energy=0.5, danceability=0.9)

    assert ranked_titles(user, [loser, winner]) == ["Zebra Winner", "Apple Loser"]


def test_rung_two_closest_energy_breaks_a_score_tie():
    """Equal scores, different distance from the target: closest wins.

    Both songs total 2.0 by different routes — one earns a genre point and loses
    energy points, the other has no genre match but sits exactly on target. Rule
    2 prefers the one that actually matches the requested energy.
    """
    user = UserProfile(favorite_genre="pop", target_energy=0.5)
    far = tie_song(1, "Far But Pop", genre="pop", energy=0.0)  # 1.0 + 1.0 = 2.0
    near = tie_song(2, "On Target", genre="rock", energy=0.5)  # 0.0 + 2.0 = 2.0

    scores = [s for _, s, _ in recommend_songs(user, [far, near], 2, TIE_WEIGHTS)]
    assert scores[0] == pytest.approx(scores[1])  # genuinely tied on rule 1
    assert ranked_titles(user, [far, near]) == ["On Target", "Far But Pop"]


def test_rung_three_danceability_breaks_a_score_and_energy_tie():
    """Same score, same energy gap: the more danceable song wins."""
    user = UserProfile(target_energy=0.5)
    dull = tie_song(1, "Aardvark Dull", energy=0.5, danceability=0.2)
    lively = tie_song(2, "Zulu Lively", energy=0.5, danceability=0.9)

    assert ranked_titles(user, [dull, lively]) == ["Zulu Lively", "Aardvark Dull"]


def test_rung_four_alphabetical_title_breaks_remaining_ties():
    """Identical on every numeric rule: fall back to title order, A first.

    Note this rung is NOT negated in the key. Alphabetical means smaller-first,
    which is already what ascending sort does — the reason the key tuple negates
    scores instead of using reverse=True.
    """
    user = UserProfile(target_energy=0.5)
    zebra = tie_song(1, "Zebra", energy=0.5, danceability=0.5)
    apple = tie_song(2, "Apple", energy=0.5, danceability=0.5)

    assert ranked_titles(user, [zebra, apple]) == ["Apple", "Zebra"]


def test_alphabetical_tie_break_ignores_capitalisation():
    """'apple' and 'Apple' should sort together, not with all capitals first.

    Comparing raw strings would put every capitalised title ahead of every
    lowercase one, because 'Z' < 'a' in character order. Lowercasing avoids that.
    """
    user = UserProfile(target_energy=0.5)
    lower = tie_song(1, "apple pie", energy=0.5, danceability=0.5)
    upper = tie_song(2, "Banana Bread", energy=0.5, danceability=0.5)

    assert ranked_titles(user, [upper, lower]) == ["apple pie", "Banana Bread"]


def test_rung_five_id_guarantees_a_total_order():
    """Two songs identical in every rule including title: lowest id wins.

    Without this last rung the order would depend on input order. The loader
    rejects duplicate ids, so this rule can always settle the contest.
    """
    user = UserProfile(target_energy=0.5)
    second = tie_song(2, "Same Title", energy=0.5, danceability=0.5)
    first = tie_song(1, "Same Title", energy=0.5, danceability=0.5)

    ranked = recommend_songs(user, [second, first], 2, TIE_WEIGHTS)
    assert [song.id for song, _, _ in ranked] == [1, 2]


def test_ranking_is_independent_of_catalog_order(catalog):
    """THE POINT OF THIS STEP.

    Shuffling the input must not change the output. Previously it could: ties
    were settled by whichever song appeared first in the CSV, because Python's
    stable sort preserves input order for equal keys. Reordering data/songs.csv
    would silently reorder recommendations.

    Now the ladder decides every position, so the ranking is a property of the
    data and the profile — not of how the file happens to be sorted.
    """
    profile = UserProfile(
        favorite_genre="rock", favorite_mood="intense", target_energy=0.9
    )
    expected = [s.title for s, _, _ in recommend_songs(profile, catalog, 19)]

    reversed_catalog = list(reversed(catalog))
    rotated_catalog = catalog[7:] + catalog[:7]

    assert [s.title for s, _, _ in recommend_songs(profile, reversed_catalog, 19)] == expected
    assert [s.title for s, _, _ in recommend_songs(profile, rotated_catalog, 19)] == expected


def test_ranking_is_repeatable_across_calls(catalog):
    """The same inputs always give the same answer.

    Determinism is what makes the evaluation scenarios in Module 4 meaningful: a
    metric that changes between runs measures nothing.
    """
    profile = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.9)

    first = [s.title for s, _, _ in recommend_songs(profile, catalog, 19)]
    second = [s.title for s, _, _ in recommend_songs(profile, catalog, 19)]

    assert first == second
