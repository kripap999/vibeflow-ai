"""Tests for the music recommender.

This file has two kinds of test.

1. The two original starter tests, which exercise the OBJECT path
   (Song / UserProfile / Recommender).

2. CHARACTERIZATION tests, added before refactoring. A characterization test
   records what the code does *today*, not what it ideally should do. Its job is
   to go red if a refactor accidentally changes behavior. Think of it as a net
   under a trapeze: it does not make the trick better, it makes falling safe.

A few tests are explicitly marked KNOWN DEFECT. Those pin down behavior we have
already agreed is wrong. We keep them so the bug is visible and documented, and
so that when we fix it the test fails *on purpose* — telling us the change
landed exactly where we intended.
"""

from pathlib import Path

import pytest

from src.recommender import (
    Recommender,
    Song,
    UserProfile,
    load_songs,
    recommend_songs,
    score_song,
)

# Build the CSV path from this file's location rather than hard-coding
# "data/songs.csv". That relative string only works if pytest happens to be run
# from the repository root; this works no matter where you run it from.
REPO_ROOT = Path(__file__).resolve().parent.parent
SONGS_CSV = str(REPO_ROOT / "data" / "songs.csv")

# The four taste profiles used below mirror src/main.py, so the numbers asserted
# here are the same numbers printed by the real application.
POP_PROFILE = {"genre": "pop", "mood": "happy", "energy": 0.9}
LOFI_PROFILE = {"genre": "lofi", "mood": "chill", "energy": 0.35, "likes_acoustic": True}

EXPECTED_COLUMNS = {
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


def song_from_row(row: dict) -> Song:
    """Convert one dict from load_songs() into a Song object.

    This bridge exists only so the equivalence tests below can feed the *same*
    catalog row into both code paths. Its existence is itself the evidence that
    we currently have two representations of a song. In the next refactor this
    conversion moves into load_songs() and this helper disappears.
    """
    return Song(
        id=row["id"],
        title=row["title"],
        artist=row["artist"],
        genre=row["genre"],
        mood=row["mood"],
        energy=row["energy"],
        tempo_bpm=row["tempo_bpm"],
        valence=row["valence"],
        danceability=row["danceability"],
        acousticness=row["acousticness"],
    )


@pytest.fixture
def catalog() -> list:
    """The real 19-song catalog, loaded once per test that asks for it."""
    return load_songs(SONGS_CSV)


@pytest.fixture
def by_title(catalog) -> dict:
    """The catalog keyed by song title, for readable lookups in assertions."""
    return {row["title"]: row for row in catalog}


# --------------------------------------------------------------------------
# Original starter tests (object path)
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


def test_load_songs_returns_plain_dicts(catalog):
    """Documents the current representation: dicts, not Song objects.

    This is the duplication we are about to remove. When load_songs starts
    returning Song objects, THIS TEST SHOULD FAIL — that failure is the signal
    that the refactor did what we wanted.
    """
    assert all(isinstance(row, dict) for row in catalog)


def test_load_songs_converts_text_to_numbers(by_title):
    """CSV files store everything as text; the loader must convert to numbers.

    Without this, "0.82" stays a string and `abs(song - target)` would explode.
    Note tempo_bpm becomes a float (118.0), not an int.
    """
    row = by_title["Sunrise City"]

    assert isinstance(row["id"], int)
    assert row["id"] == 1

    for field in ("energy", "tempo_bpm", "valence", "danceability", "acousticness"):
        assert isinstance(row[field], float), f"{field} should be a float"

    assert row["energy"] == pytest.approx(0.82)
    assert row["tempo_bpm"] == pytest.approx(118.0)

    # Text fields stay text.
    assert row["title"] == "Sunrise City"
    assert row["artist"] == "Neon Echo"


def test_load_songs_preserves_csv_order(catalog):
    """Row order is preserved. This matters more than it looks.

    Python's sort is 'stable': items with equal scores keep their original
    relative order. So today the CSV order silently acts as our tie-breaker.
    """
    assert catalog[0]["title"] == "Sunrise City"
    assert catalog[-1]["title"] == "Dust Road Home"


def test_load_songs_every_row_has_all_columns(catalog):
    """Every song carries the full set of fields — no ragged rows."""
    for row in catalog:
        assert set(row.keys()) == EXPECTED_COLUMNS, f"bad columns in {row}"


def test_catalog_ids_are_unique(catalog):
    """A data guard, not a characterization test.

    Nothing in the code enforces unique IDs yet. This test asserts the data file
    is currently clean, so if someone pastes in a duplicate ID while expanding
    the catalog to 75+ tracks, we hear about it immediately.
    """
    ids = [row["id"] for row in catalog]
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

    This is the case your model card calls out: a pop song with the wrong mood
    still ranks well because two of three signals agree. Worth pinning, because
    it is the behavior we may deliberately change later.
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


def test_score_song_with_no_preferences_scores_zero(by_title):
    """No stated preferences means no points and no reasons.

    Every scoring rule is guarded by `if "..." in user_prefs`, so an empty
    profile skips all of them rather than crashing.
    """
    score, reasons = score_song({}, by_title["Sunrise City"])

    assert score == 0.0
    assert reasons == []


def test_score_song_ignores_missing_preference_keys(by_title):
    """Asking about genre alone must not require the other keys to be present."""
    score, reasons = score_song({"genre": "pop"}, by_title["Sunrise City"])

    assert score == pytest.approx(2.0)
    assert len(reasons) == 1


def test_score_song_energy_reward_is_graded_not_all_or_nothing(by_title):
    """Energy is scored by CLOSENESS, so it pays out on a sliding scale.

    Exact match  -> the full 2.0
    Far from target -> a small fraction

    This is the most important idea in the current scorer and the one that
    carries straight into VibeFlow's journey planning: we reward songs NEAR a
    target energy, not simply the loudest songs available.
    """
    song = by_title["Sunrise City"]  # energy 0.82

    exact, _ = score_song({"energy": 0.82}, song)
    far, _ = score_song({"energy": 0.0}, song)

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

    assert score_song({"likes_acoustic": False}, electronic)[0] == pytest.approx(1.0)
    assert score_song({"likes_acoustic": True}, electronic)[0] == pytest.approx(0.0)
    assert score_song({"likes_acoustic": True}, acoustic)[0] == pytest.approx(1.0)
    assert score_song({"likes_acoustic": False}, acoustic)[0] == pytest.approx(0.0)


def test_score_song_currently_allows_negative_scores_KNOWN_DEFECT():
    """KNOWN DEFECT: out-of-range data produces a nonsense negative score.

    The energy term is 2.0 * (1 - |song - target|). That formula only behaves if
    both values sit inside 0.0-1.0. Feed it energy=5.0 and it returns -8.0, plus
    the malformed explanation text '(+-8.00)'. No exception is raised, so a
    single CSV typo would corrupt rankings silently.

    We pin it here so the bug is documented. When we add validation, this test
    should be REPLACED by one asserting a clear error is raised.
    """
    score, reasons = score_song({"energy": 0.0}, {"energy": 5.0})

    assert score == pytest.approx(-8.0)
    assert "(+-8.00)" in reasons[0]


# --------------------------------------------------------------------------
# recommend_songs: ranking and top-k
# --------------------------------------------------------------------------

def test_recommend_songs_returns_exactly_k_results(catalog):
    assert len(recommend_songs(POP_PROFILE, catalog, k=5)) == 5
    assert len(recommend_songs(POP_PROFILE, catalog, k=1)) == 1


def test_recommend_songs_returns_song_score_explanation_triples(catalog):
    """Documents the return SHAPE: a 3-tuple of (dict, float, str).

    src/main.py unpacks exactly this shape, so anything that changes it breaks
    the application.
    """
    results = recommend_songs(POP_PROFILE, catalog, k=3)

    for song, score, explanation in results:
        assert isinstance(song, dict)
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
    titles = [song["title"] for song, _, _ in recommend_songs(POP_PROFILE, catalog, k=3)]

    assert titles == ["Sunrise City", "Rooftop Lights", "Gym Hero"]


def test_recommend_songs_does_not_modify_the_catalog(catalog):
    """Ranking must not reorder or damage the caller's list.

    recommend_songs builds a new list and sorts that, leaving the input alone.
    A function that quietly rearranges its argument causes bugs that are very
    hard to trace, so this is worth locking down.
    """
    before = [row["title"] for row in catalog]

    recommend_songs(POP_PROFILE, catalog, k=5)

    assert [row["title"] for row in catalog] == before


def test_recommend_songs_explanation_matches_score_reasons(catalog, by_title):
    """The printed explanation is the scorer's reasons joined by '; '.

    Confirms nothing is added or dropped between scoring and display.
    """
    results = recommend_songs(POP_PROFILE, catalog, k=1)
    song, _, explanation = results[0]

    _, reasons = score_song(POP_PROFILE, by_title[song["title"]])
    assert explanation == "; ".join(reasons)


def test_ties_are_currently_broken_by_float_noise_KNOWN_DEFECT(catalog):
    """KNOWN DEFECT: tied songs are ordered by floating-point rounding.

    For the rock profile, two songs both display 1.84:

        Sunrise City -> 1.8399999999999999
        Iron Verdict -> 1.84

    Both come from 2.0 * (1 - 0.08), but |0.82-0.9| and |0.98-0.9| land on
    different binary floats. So Iron Verdict wins by a difference in the 16th
    decimal place — an ordering no user could ever be given a reason for.

    Reproducible, but arbitrary. When we add real tie-breaking rules this test
    should fail and be replaced.
    """
    rock_profile = {"genre": "rock", "mood": "intense", "energy": 0.9}
    ranked = recommend_songs(rock_profile, catalog, k=19)
    titles = [song["title"] for song, _, _ in ranked]

    assert titles.index("Iron Verdict") < titles.index("Sunrise City")


# --------------------------------------------------------------------------
# Equivalence: the two code paths must agree
# --------------------------------------------------------------------------

def test_dict_path_and_object_path_give_identical_scores(by_title):
    """THE KEY TEST OF THIS STEP.

    The application scores dicts via score_song. The tests score Song objects
    via Recommender._score. Both must produce the same number for the same song.

    This is what makes the coming consolidation provably behavior-preserving: if
    the two paths agree now, and still agree after we delete one of them, then
    we removed duplication without changing results.
    """
    row = by_title["Sunrise City"]

    prefs = {"genre": "pop", "mood": "happy", "energy": 0.9, "likes_acoustic": False}
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.9,
        likes_acoustic=False,
    )

    dict_score, dict_reasons = score_song(prefs, row)
    object_score, object_reasons = Recommender([])._score(user, song_from_row(row))

    assert dict_score == pytest.approx(object_score)
    assert dict_reasons == object_reasons
    assert dict_score == pytest.approx(7.84)


def test_both_paths_rank_the_full_catalog_identically(catalog):
    """The two RANKERS must also agree, not just the two scorers.

    recommend_songs and Recommender.recommend are separate implementations of
    'sort by score and take the top k'. Pinning that they produce the same order
    over all 19 real songs is what protects us when we collapse them into one.
    """
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.9, "likes_acoustic": False}
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.9,
        likes_acoustic=False,
    )

    dict_titles = [song["title"] for song, _, _ in recommend_songs(prefs, catalog, k=5)]

    songs = [song_from_row(row) for row in catalog]
    object_titles = [song.title for song in Recommender(songs).recommend(user, k=5)]

    assert dict_titles == object_titles


def test_explain_recommendation_adds_a_score_prefix(by_title):
    """Documents the one real difference between the two paths.

    recommend_songs returns only the joined reasons:
        "mood match: happy (+3.0); genre match: pop (+2.0); ..."

    Recommender.explain_recommendation prepends the total:
        "Score 7.84 — mood match: happy (+3.0); ..."

    Two explanation formats for the same facts. Pinned here so we choose one
    deliberately during consolidation instead of losing one by accident.
    """
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.9,
        likes_acoustic=False,
    )
    song = song_from_row(by_title["Sunrise City"])

    explanation = Recommender([song]).explain_recommendation(user, song)

    assert explanation.startswith("Score 7.84 — ")
    assert "mood match: happy (+3.0)" in explanation
