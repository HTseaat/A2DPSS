"""Regression tests for exact PVSS verification optimizations."""

from __future__ import annotations

from beaver.chunk_proof import (
    CHALLENGE_MASK,
    NUM_CHUNKS,
    NUM_ZK_REPETITIONS,
    _derive_e,
    _hash_int,
)


def test_parallel_challenges_are_byte_exact_with_original_derivation() -> None:
    instance = {
        "version": 1,
        "rpks": ["recipient-0", "recipient-1"],
        "nested": {"z": 3, "a": [2, 1]},
    }
    y0 = "y0"
    b_list = [f"b-{index}" for index in range(NUM_ZK_REPETITIONS)]
    c_list = [f"c-{index}" for index in range(NUM_ZK_REPETITIONS)]
    seed = {
        "instance": instance,
        "y0": y0,
        "B": b_list,
        "C": c_list,
    }
    expected = [
        [
            [
                _hash_int(
                    "chunk-proof-e",
                    {"seed": seed, "i": i, "j": j, "k": k},
                )
                & CHALLENGE_MASK
                for k in range(NUM_ZK_REPETITIONS)
            ]
            for j in range(NUM_CHUNKS)
        ]
        for i in range(len(instance["rpks"]))
    ]

    assert _derive_e(instance, y0, b_list, c_list) == expected
    assert _derive_e(instance, y0, b_list, c_list, parallel=True) == expected
