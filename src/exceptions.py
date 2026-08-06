"""Exception types for VibeFlow.

Defining our own exceptions rather than raising bare `ValueError` everywhere has
two benefits:

1. A caller can catch exactly the failure it knows how to handle. `except
   CatalogError` catches a bad data file without also swallowing an unrelated
   arithmetic bug.
2. Every VibeFlow failure shares a common base, so an outer layer (a CLI, or
   later a Streamlit app) can catch `VibeFlowError` to show a friendly message
   while letting genuine programming bugs crash loudly and visibly.

That second point is the important one. Catching bare `Exception` hides real
bugs; catching nothing means users see a stack trace. A base class for *our*
errors gives us a middle path.
"""


class VibeFlowError(Exception):
    """Base class for every error VibeFlow raises on purpose."""


class InvalidSongError(VibeFlowError):
    """One song's data is invalid — a bad number, a blank title, a range error.

    Raised by `Song.__post_init__`, so it is impossible to construct a Song that
    violates its own invariants, whether the data came from a CSV or from code.
    """


class CatalogError(VibeFlowError):
    """The song catalog as a whole cannot be used.

    Covers problems that no single song can detect on its own: a missing column,
    duplicate IDs across rows, an unreadable file, or an empty catalog. Its
    message aggregates every problem found so one run tells you everything that
    needs fixing.
    """
