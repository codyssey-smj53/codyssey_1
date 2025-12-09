"""Utility to maximize representable integers with three dice.

Given two fixed dice (A and B) whose six faces contain digits in the range 0-9,
this module determines how to label the six faces of a configurable die C so
that the count of three-digit numbers in a target range that can be displayed by
arranging the dice is maximized.

Numbers below 100 are written with leading zeros (e.g. 007). Digits 6 and 9 are
interchangeable on any face because rotating a die by 180° swaps them.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, List, Sequence, Tuple

DIGIT_DOMAIN: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8)
DIGIT_TO_INDEX = {digit: idx for idx, digit in enumerate(DIGIT_DOMAIN)}
DIGIT_TO_INDEX[9] = DIGIT_TO_INDEX[6]
_ACTIVE_BOUNDS: Tuple[int, int] | None = None


@dataclass(slots=True)
class OptimizationResult:
    """Result of optimizing die C."""

    max_count: int
    c_faces: List[int]
    covered_values: List[int]


def _normalize_digit(digit: int) -> int:
    return 6 if digit == 9 else digit


def _build_die_mask(digits: Sequence[int]) -> List[bool]:
    if len(digits) != 6:
        raise ValueError("Each die must provide exactly 6 digits.")
    mask = [False] * 10
    for digit in digits:
        if not 0 <= digit <= 9:
            raise ValueError("Digits must be within [0, 9].")
        if digit in (6, 9):
            mask[6] = True
            mask[9] = True
        else:
            mask[digit] = True
    return mask


def _digit_available(mask: Sequence[bool], digit: int) -> bool:
    return mask[digit]


def _need_masks(
    a_mask: Sequence[bool], b_mask: Sequence[bool], lower: int, upper: int
) -> Tuple[List[int], List[int]]:
    needs: List[int] = []
    values: List[int] = []
    for value in range(lower, upper + 1):
        digits = [int(ch) for ch in f"{value:03d}"]
        need_mask = 0
        for c_idx in range(3):
            c_digit = digits[c_idx]
            other_indices = [idx for idx in range(3) if idx != c_idx]
            first, second = digits[other_indices[0]], digits[other_indices[1]]
            if (
                _digit_available(a_mask, first)
                and _digit_available(b_mask, second)
            ) or (
                _digit_available(a_mask, second)
                and _digit_available(b_mask, first)
            ):
                norm = _normalize_digit(c_digit)
                need_mask |= 1 << DIGIT_TO_INDEX[norm]
        needs.append(need_mask)
        values.append(value)
    return needs, values


def _subset_to_digits(subset_mask: int) -> Tuple[int, ...]:
    digits: List[int] = []
    for idx, digit in enumerate(DIGIT_DOMAIN):
        if subset_mask & (1 << idx):
            digits.append(digit)
    return tuple(digits)


def _select_best_subset(need_masks: Sequence[int]) -> Tuple[int, int]:
    best_mask = 0
    best_count = -1
    best_bits = -1
    best_digits_tuple: Tuple[int, ...] = ()

    total_subsets = 1 << len(DIGIT_DOMAIN)
    for subset in range(1, total_subsets):
        bits = subset.bit_count()
        if bits > 6:
            continue
        count = 0
        for need_mask in need_masks:
            if need_mask & subset:
                count += 1
        if count == 0 and best_count == -1:
            # Defer selecting a zero-count set until we see if any positive set exists.
            pass
        if (count > best_count) or (
            count == best_count
            and (
                bits > best_bits
                or (
                    bits == best_bits
                    and _subset_to_digits(subset) < best_digits_tuple
                )
            )
        ):
            best_count = count
            best_mask = subset
            best_bits = bits
            best_digits_tuple = _subset_to_digits(subset)

    if best_mask == 0:
        # No subset improved the initial state; fall back to the first six digits.
        best_mask = sum(1 << idx for idx in range(6))
        best_count = 0
    return best_mask, best_count


def _mask_to_faces(mask: int) -> List[int]:
    digits = list(_subset_to_digits(mask))
    if not digits:
        digits = list(DIGIT_DOMAIN[:6])
    faces = digits.copy()
    idx = 0
    while len(faces) < 6:
        faces.append(digits[idx % len(digits)])
        idx += 1
    return faces[:6]


def maximize_representable_numbers(
    a_digits: Sequence[int],
    b_digits: Sequence[int],
    lower: int,
    upper: int,
) -> OptimizationResult:
    """Compute the optimal configuration for die C.

    Args:
        a_digits: Six digits describing die A.
        b_digits: Six digits describing die B.
        lower: Smallest value in the interval (inclusive).
        upper: Largest value in the interval (inclusive).

    Returns:
        An OptimizationResult containing the maximum count, the labels for die C,
        and the list of representable values.
    """

    if len(a_digits) != 6 or len(b_digits) != 6:
        raise ValueError("Dice A and B must each contain six digits.")
    if not (1 <= lower <= upper <= 999):
        raise ValueError("Range must satisfy 1 <= l <= r <= 999.")

    a_mask = _build_die_mask(a_digits)
    b_mask = _build_die_mask(b_digits)
    need_masks, values = _need_masks(a_mask, b_mask, lower, upper)
    best_mask, best_count = _select_best_subset(need_masks)
    c_faces = _mask_to_faces(best_mask)
    covered_values = [
        value for value, need_mask in zip(values, need_masks) if need_mask & best_mask
    ]
    return OptimizationResult(best_count, c_faces, covered_values)


def func(dice_a: Sequence[int], dice_b: Sequence[int]) -> OptimizationResult:
    """Wrapper that reuses the bounds captured by ``main``."""

    if _ACTIVE_BOUNDS is None:
        raise RuntimeError("Bounds are not initialized. Run main() first.")
    lower, upper = _ACTIVE_BOUNDS
    return maximize_representable_numbers(dice_a, dice_b, lower, upper)


def _parse_digits_from_line(line: str) -> List[int]:
    extracted = [int(ch) for ch in re.findall(r"\d", line)]
    if len(extracted) < 6:
        raise ValueError("Expected at least six digits per line for a die.")
    return extracted[:6]


def _parse_bounds(lines: Iterable[str]) -> Tuple[int, int]:
    tokens: List[int] = []
    for line in lines:
        tokens.extend(int(num) for num in re.findall(r"-?\d+", line))
    if len(tokens) < 2:
        raise ValueError("Expected two integers describing the range [l, r].")
    return tokens[0], tokens[1]


def _format_values(values: Sequence[int]) -> str:
    return " ".join(f"{value:03d}" for value in values)


def main() -> None:
    import sys

    raw = sys.stdin.read().strip()
    if not raw:
        print(
            "Provide three lines: die A, die B, then the l r bounds (with digits).",
            file=sys.stderr,
        )
        return
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) < 3:
        raise SystemExit("Need at least three non-empty lines of input.")

    a_digits = _parse_digits_from_line(lines[0])
    b_digits = _parse_digits_from_line(lines[1])
    lower, upper = _parse_bounds(lines[2:])

    global _ACTIVE_BOUNDS
    _ACTIVE_BOUNDS = (lower, upper)

    result = func(a_digits, b_digits)
    print(f"max_count: {result.max_count}")
    print("c_faces: " + " ".join(str(face) for face in result.c_faces))
    print("covered_values: " + _format_values(result.covered_values))


if __name__ == "__main__":
    main()
