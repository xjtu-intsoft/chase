from dataclasses import dataclass
from typing import List, Callable, Any, Generic, TypeVar

from duorat.asdl.asdl_ast import AbstractSyntaxTree


B = TypeVar("B")
T = TypeVar("T")


@dataclass
class Hypothesis(Generic[B]):
    beam_builder: B
    tokens: list
    scores: List[float]  # past total scores (at each decoding time step)
    score: float = 0.0  # total score

    def is_finished(self) -> bool:
        raise NotImplementedError


@dataclass
class Candidate(Generic[T]):
    token: T  # candidate token to add
    score: float  # total score
    prev_hypothesis: Hypothesis


@dataclass
class FinishedBeam(object):
    ast: AbstractSyntaxTree
    score: float


def beam_search(
    initial_hypothesis: Hypothesis[B],
    beam_size: int,
    max_steps: int,
    get_new_hypothesis: Callable[[Candidate[T]], Hypothesis[B]],
    get_continuations: Callable[[List[Hypothesis[B]], int], List[Candidate[T]]],
) -> List[Hypothesis[B]]:
    """
    initial hypothesis:
    get_new_hypothesis: get new hypothesis from candidate
    get_continuations: get list of possible continuations (candidates) from hypothesis
    """
    beam: List[Hypothesis] = [initial_hypothesis]
    finished = []
    for step in range(max_steps):
        # Stop if all beams are finished
        if len(finished) == beam_size:
            break

        # Get possible continuations
        candidates = get_continuations(beam, step)

        # Keep the top K expansions
        candidates.sort(key=lambda c: c.score, reverse=True)
        candidates = candidates[: beam_size - len(finished)]

        # Create the new hypotheses from the expansions
        beam = []
        for candidate in candidates:
            new_hyp = get_new_hypothesis(candidate)
            if new_hyp.is_finished():
                finished.append(new_hyp)
            else:
                beam.append(new_hyp)

    finished.sort(key=lambda h: h.score, reverse=True)
    return finished
