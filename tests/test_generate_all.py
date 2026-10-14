from scripts.generate_all import seed_for_alloy


def test_seed_for_alloy_uses_distinct_reproducible_ranges() -> None:
    base_seed = 20260721
    count_per_alloy = 32

    seeds = [
        seed_for_alloy(base_seed, alloy_id, count_per_alloy)
        for alloy_id in range(1, 23)
    ]

    assert seeds[0] == base_seed
    assert seeds[1] == base_seed + count_per_alloy
    assert seeds[-1] == base_seed + 21 * count_per_alloy
    assert len(set(seeds)) == 22
