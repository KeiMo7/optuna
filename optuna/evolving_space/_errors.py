from __future__ import annotations


class SearchSpaceConflictError(ValueError):
    """Raised when a parameter changes its type in a lossless-incompatible way.

    The message names the conflicting parameter, the distribution recorded in
    history and the newly requested distribution.
    """


class SearchSpaceNoOverlapError(RuntimeError):
    """Raised when the new search space shares no usable dimension with history.

    Continuing optimization in this state would make every new trial
    uninformative for the sampler, so the transition is refused with a readable
    explanation instead.
    """
