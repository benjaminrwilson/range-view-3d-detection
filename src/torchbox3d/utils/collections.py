"""Methods to help manipulate collections."""

from typing import Any, Iterator, Sequence


def flatten(elems: Sequence[Any]) -> Iterator[Any]:
    """Recursively flattens a nested collection of sequences.

    Args:
        elems: A nested list of sequences.

    Yields:
        The recursively-flattened sequence.
    """
    for elem in elems:
        if isinstance(elem, Sequence) and not isinstance(elem, (str, bytes)):
            yield from flatten(elem)
        else:
            yield elem
