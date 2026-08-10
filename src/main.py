"""Command line runner for the Music Recommender Simulation.

Loads the song catalog and prints ranked recommendations for several taste
profiles, each with a plain-language reason for every pick.

Run from the repository root:

    python -m src.main
"""

from src.models import UserProfile
from src.recommender import load_songs, recommend_songs

# A set of taste profiles used to stress-test the recommender.
# The last one is "adversarial": high energy paired with a sad mood, which
# almost never occur together in real music, to see how scoring copes.
#
# Three profiles leave `likes_acoustic` unset. That is not an oversight — an
# unstated preference is skipped by the scorer, whereas stating False would earn
# a +1.0 agreement point on every non-acoustic song and inflate the scores.
PROFILES = [
    (
        "High-Energy Pop",
        UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.9),
    ),
    (
        "Chill Lofi",
        UserProfile(
            favorite_genre="lofi",
            favorite_mood="chill",
            target_energy=0.35,
            likes_acoustic=True,
        ),
    ),
    (
        "Deep Intense Rock",
        UserProfile(favorite_genre="rock", favorite_mood="intense", target_energy=0.9),
    ),
    (
        "Adversarial: Sad but High-Energy",
        UserProfile(favorite_genre="metal", favorite_mood="sad", target_energy=0.9),
    ),
]


def describe_profile(user: UserProfile) -> str:
    """Render only the preferences the listener actually stated.

    Note the `is not None` check rather than a plain truthiness test. Writing
    `if value` would hide `likes_acoustic=False` and `target_energy=0.0`, since
    both are falsy but were genuinely stated.
    """
    stated = {
        "genre": user.favorite_genre,
        "mood": user.favorite_mood,
        "energy": user.target_energy,
        "likes_acoustic": user.likes_acoustic,
    }
    inner = ", ".join(f"{key!r}: {value!r}" for key, value in stated.items() if value is not None)
    return "{" + inner + "}"


def print_recommendations(name: str, user: UserProfile, songs: list, k: int = 5) -> None:
    """Print the top k recommendations for one named profile."""
    print(f"\n=== {name} ===")
    print(f"User profile: {describe_profile(user)}\n")
    for rank, (song, score, explanation) in enumerate(recommend_songs(user, songs, k), start=1):
        print(f"{rank}. {song.title} by {song.artist}  (Score: {score:.2f})")
        print(f"   Because: {explanation}")


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    for name, user in PROFILES:
        print_recommendations(name, user, songs)


if __name__ == "__main__":
    main()
