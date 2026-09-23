"""Build deterministic win44 subsets without changing the original loaders.

Run with a Python environment containing numpy, xarray, torch and a NetCDF
backend. Use --check to regenerate in memory and compare existing JSON bytes.
Use --overwrite to replace existing subsets after an intentional design change.
No training, preprocessing, normalization, or existing-source writes occur.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import importlib
import json
from pathlib import Path, PureWindowsPath
import random
import re
import sys

# Importing the original package must not create/update its __pycache__.
sys.dont_write_bytecode = True

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODE = ROOT / "src"
LEADS = (7, 14)
SEEDS = (42, 7, 123)
WINDOW = 44
BLOCK_DAYS = 90
RATIOS = (100, 50, 25, 10)
SEASONS = ("Winter", "Spring", "Summer", "Autumn")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def date_text(dates: np.ndarray, index: int) -> str:
    return str(dates[int(index)])


def span(dates: np.ndarray, first: int, last: int) -> dict:
    return {"start_index": int(first), "end_index": int(last),
            "start": date_text(dates, first), "end": date_text(dates, last),
            "days": int(last - first + 1)}


def split_details(dates: np.ndarray, indices: np.ndarray, history: int, lead: int) -> dict:
    first, last = int(indices[0]), int(indices[-1])
    return {
        "sample_count": len(indices),
        "start_indices": [first, last],
        "sample_start_dates": [date_text(dates, first), date_text(dates, last)],
        "input_dates": [date_text(dates, first), date_text(dates, last + history - 1)],
        "target_dates": [date_text(dates, first + history - 1 + lead),
                         date_text(dates, last + history - 1 + lead)],
        "sample_footprint": span(dates, first, last + history - 1 + lead),
    }


def load_context(args: argparse.Namespace):
    code = args.code_dir.resolve()
    manifest_path = (args.manifest or code.parent / "outputs" / "main_experiment_final"
                     / "experiment_manifest_final.json").resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    runner_path = code / "run_main_experiment_final.ps1"
    runner = runner_path.read_text(encoding="utf-8-sig")
    history = int(manifest["history_days"])
    history_match = re.search(r"\[int\]\$History\s*=\s*(\d+)", runner)
    assert history_match and int(history_match[1]) == history == 30
    assert WINDOW == history + max(LEADS)
    assert manifest["final_physics"] == "P0"
    assert all(lead in manifest["leads"] for lead in LEADS)
    data_name = PureWindowsPath(manifest["data_path"]).name
    assert data_name in runner, "Runner and manifest dataset names differ"
    # The original D:/MHW project is relocated to code.parent (the workspace).
    data_path = (args.data_path or code.parent / "data" / data_name).resolve()
    assert data_path.name == data_name, "Use the final experiment's dataset"

    sys.path.insert(0, str(code))
    datasets = importlib.import_module("mhw.datasets")
    assert Path(datasets.__file__).resolve() == code / "mhw" / "datasets.py"
    # Deliberately use the original function, without reimplementing its split.
    make_time_splits = datasets.make_time_splits
    with xr.open_dataset(data_path) as ds:
        dates = ds["time"].values.astype("datetime64[D]")
    assert not np.isnat(dates).any()
    assert np.all(np.diff(dates) == np.timedelta64(1, "D")), "Expected daily contiguous time"
    splits = {lead: make_time_splits(len(dates), history, lead) for lead in LEADS}
    candidates = np.intersect1d(splits[7].train, splits[14].train)
    assert len(candidates) and np.all(np.diff(candidates) == 1)
    details = {
        str(lead): {name: split_details(dates, getattr(split, name), history, lead)
                    for name in ("train", "val", "test")}
        for lead, split in splits.items()
    }
    # Common calendar footprint of the two original training splits. Taking the
    # earlier end makes every complete 44-day window stay inside BOTH footprints.
    first = max(int(split.train[0]) for split in splits.values())
    last = min(int(split.train[-1]) + history - 1 + lead for lead, split in splits.items())
    metadata = {
        "schema_version": 2,
        "source": {
            "code_folder": code.relative_to(ROOT).as_posix() if code.is_relative_to(ROOT) else str(code),
            "split_function": "mhw.datasets.make_time_splits",
            "split_source_sha256": sha256(code / "mhw" / "datasets.py"),
            "runner_sha256": sha256(runner_path),
            "manifest_sha256": sha256(manifest_path),
            "manifest_data_path": manifest["data_path"],
            "resolved_data_path": data_path.relative_to(ROOT).as_posix() if data_path.is_relative_to(ROOT) else str(data_path),
            "data_sha256": sha256(data_path),
        },
        "history_days": history, "leads": list(LEADS), "window_days": WINDOW,
        "block_days": BLOCK_DAYS, "starts_per_block": BLOCK_DAYS - WINDOW + 1,
        "dataset": span(dates, 0, len(dates) - 1),
        "original_splits": details,
        "candidate_starts": {**span(dates, int(candidates[0]), int(candidates[-1])),
                             "count": len(candidates)},
        "blocking_training_span": span(dates, first, last),
        "training_span_rule": "Intersection of train sample calendar footprints (input start through target), not just sample-start dates",
        "rounding": "Python round(percent * total_blocks / 100), ties to even",
        "coverage_definition": "Union of all 44-day sample windows; denominator is blocking_training_span.days",
        "season_definition": "Block midpoint start + 44 days (earlier central day of a 90-day block)",
        "ranking": "One seeded base season order, left-rotated by period index 0..3; shuffled blocks within seasons; season round-robin then fixed shuffled period round-robin; ratio selections are prefixes",
    }
    return dates, splits, candidates, metadata


def make_blocks(dates: np.ndarray, candidates: np.ndarray, metadata: dict):
    train_span = metadata["blocking_training_span"]
    first, last = train_span["start_index"], train_span["end_index"]
    candidate_set = set(map(int, candidates))
    blocks = []
    for block_id, start in enumerate(range(first, last - BLOCK_DAYS + 2, BLOCK_DAYS)):
        end = start + BLOCK_DAYS - 1
        starts = list(range(start, end - WINDOW + 2))
        assert len(starts) == 47 and set(starts) <= candidate_set
        center = date_text(dates, start + (BLOCK_DAYS - 1) // 2)
        month = int(center[5:7])
        season = SEASONS[(month % 12) // 3]
        blocks.append({"block_id": block_id, "start": date_text(dates, start),
                       "end": date_text(dates, end), "start_index": start,
                       "end_index": end, "center_date": center, "season": season})
    assert len(blocks) >= 4
    periods = []
    for i, ids in enumerate(np.array_split(np.arange(len(blocks)), 4), 1):
        period = f"P{i}"
        for block_id in ids:
            blocks[int(block_id)]["interval"] = period
        periods.append({"interval": period, "block_count": len(ids),
                        "block_ids": list(map(int, ids)),
                        "start": blocks[int(ids[0])]["start"],
                        "end": blocks[int(ids[-1])]["end"]})
    discarded_first = blocks[-1]["end_index"] + 1
    metadata.update({"total_blocks": len(blocks), "intervals": periods,
                     "discarded_tail": span(dates, discarded_first, last)
                     if discarded_first <= last else None})
    return blocks


def rank_blocks(blocks: list[dict], seed: int):
    rng = random.Random(seed)
    base_seasons = list(SEASONS)
    rng.shuffle(base_seasons)
    queues = {}
    season_orders = {}
    for k, period in enumerate(("P1", "P2", "P3", "P4")):
        buckets = {season: [b["block_id"] for b in blocks
                            if b["interval"] == period and b["season"] == season]
                   for season in SEASONS}
        for bucket in buckets.values():
            rng.shuffle(bucket)
        # Stagger the same cycle so each season occurs twice in the first two
        # picks of all four periods combined (the eight-block 10% subset).
        seasons = base_seasons[k:] + base_seasons[:k]
        season_orders[period] = seasons
        ranked = []
        while any(buckets.values()):
            for season in seasons:
                if buckets[season]:
                    ranked.append(buckets[season].pop())
        queues[period] = deque(ranked)
    period_order = list(queues)
    rng.shuffle(period_order)
    ranking = []
    while any(queues.values()):
        for period in period_order:
            if queues[period]:
                ranking.append(queues[period].popleft())
    assert len(ranking) == len(set(ranking)) == len(blocks)
    return ranking, period_order, season_orders


def make_payload(dates, splits, candidates, blocks, metadata, seed):
    ranking, period_order, season_orders = rank_blocks(blocks, seed)
    payload = {"_metadata": {**metadata, "seed": seed, "ranked_block_ids": ranking,
                             "interval_order": period_order, "season_orders": season_orders}}
    candidate_set = set(map(int, candidates))
    per_lead = {lead: set(map(int, split.train)) for lead, split in splits.items()}
    train_span = metadata["blocking_training_span"]
    for percent in RATIOS:
        count = round(percent * len(blocks) / 100)
        selected = sorted((blocks[i] for i in ranking[:count]), key=lambda b: b["block_id"])
        starts, covered = [], set()
        for block in selected:
            block_starts = range(block["start_index"], block["end_index"] - WINDOW + 2)
            for index in block_starts:
                assert index in candidate_set
                assert train_span["start_index"] <= index <= index + WINDOW - 1 <= train_span["end_index"]
                assert block["start_index"] <= index <= index + WINDOW - 1 <= block["end_index"]
                for lead in LEADS:
                    assert index in per_lead[lead]
                    target = index + metadata["history_days"] - 1 + lead
                    assert target <= int(splits[lead].train[-1]) + metadata["history_days"] - 1 + lead
                starts.append(index)
                covered.update(range(index, index + WINDOW))
        assert len(starts) == len(set(starts)) == count * 47
        assert len(covered) == count * BLOCK_DAYS
        interval_counts = {p: sum(b["interval"] == p for b in selected) for p in period_order}
        assert max(interval_counts.values()) - min(interval_counts.values()) <= 1
        season_counts = {s: sum(b["season"] == s for b in selected) for s in SEASONS}
        cross = {p: {s: sum(b["interval"] == p and b["season"] == s for b in selected)
                     for s in SEASONS} for p in sorted(period_order)}
        if percent == 10:
            assert min(interval_counts.values()) >= 1
            assert all(count == 2 for count in season_counts.values()), (
                f"10% seasonal imbalance for seed {seed}: {season_counts}"
            )
        payload[str(percent)] = {
            "selected_blocks": selected,
            "start_dates": [date_text(dates, i) for i in starts],
            "start_indices": starts,
            "summary": {"block_count": count, "start_count": len(starts),
                        "unique_date_count": len(covered),
                        "training_date_coverage_percent": 100 * len(covered) / train_span["days"],
                        "retained_block_date_coverage_percent": 100 * len(covered) / (len(blocks) * BLOCK_DAYS),
                        "interval_counts": dict(sorted(interval_counts.items())),
                        "season_counts": season_counts, "interval_season_counts": cross},
        }
    for low, high in ((10, 25), (25, 50), (50, 100)):
        assert set(payload[str(low)]["start_indices"]) < set(payload[str(high)]["start_indices"])
        assert {b["block_id"] for b in payload[str(low)]["selected_blocks"]} < {
            b["block_id"] for b in payload[str(high)]["selected_blocks"]}
    return payload


def show_report(payload):
    print(f"\nSEED {payload['_metadata']['seed']}")
    print("ratio blocks starts unique_days coverage_train% coverage_retained%")
    for percent in RATIOS:
        s = payload[str(percent)]["summary"]
        print(f"{percent:>3}% {s['block_count']:>6} {s['start_count']:>6} {s['unique_date_count']:>11} "
              f"{s['training_date_coverage_percent']:>15.4f} {s['retained_block_date_coverage_percent']:>18.4f}")
        print("  period Winter Spring Summer Autumn Total")
        for period, counts in s["interval_season_counts"].items():
            print(f"  {period:<6} " + " ".join(f"{counts[season]:>6}" for season in SEASONS)
                  + f" {sum(counts.values()):>5}")
        print("  TOTAL  " + " ".join(f"{s['season_counts'][season]:>6}" for season in SEASONS)
              + f" {s['block_count']:>5}")
    print("PASS: nested blocks/starts, train membership, 44-day containment, balanced intervals, all intervals in 10%, exactly 2 blocks per season in 10%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, default=DEFAULT_CODE)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "subsets")
    write_mode = parser.add_mutually_exclusive_group()
    write_mode.add_argument("--check", action="store_true", help="Only compare regenerated bytes to saved files")
    write_mode.add_argument("--overwrite", action="store_true", help="Replace existing subset JSON after all validation assertions pass")
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Do not use python -O: validation assertions must run")
    dates, splits, candidates, metadata = load_context(args)
    blocks = make_blocks(dates, candidates, metadata)
    print("Original make_time_splits imported from:", metadata["source"]["code_folder"])
    for lead, parts in metadata["original_splits"].items():
        for name, info in parts.items():
            print(f"lead={lead} {name}: n={info['sample_count']} starts={info['sample_start_dates']} "
                  f"inputs={info['input_dates']} targets={info['target_dates']}")
    print("Common candidate starts:", metadata["candidate_starts"])
    print("Common training calendar span:", metadata["blocking_training_span"])
    print("Complete blocks:", len(blocks), "discarded tail:", metadata["discarded_tail"])
    print("Periods:", metadata["intervals"])
    payloads = {seed: make_payload(dates, splits, candidates, blocks, metadata, seed) for seed in SEEDS}
    for percent in RATIOS:
        assert len({len(p[str(percent)]["start_indices"]) for p in payloads.values()}) == 1
    assert all(payloads[seed]["100"] == payloads[42]["100"] for seed in SEEDS)
    for seed, payload in payloads.items():
        show_report(payload)
        encoded = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
        path = args.output_dir / f"train_subsets_v2_win44_seed{seed}.json"
        if args.check:
            assert path.read_bytes() == encoded, f"Reproducibility mismatch: {path}"
            print("PASS: byte-identical regeneration:", path.name)
        elif path.exists():
            if path.read_bytes() != encoded:
                if not args.overwrite:
                    raise FileExistsError(f"Refusing to overwrite different existing data; use --overwrite: {path}")
                path.write_bytes(encoded)
                print("Regenerated:", path.name)
            else:
                print("Already identical:", path.name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(encoded)
            print("Created:", path.name)
    print("PASS: identical sample counts across seeds; identical 100% subsets")


if __name__ == "__main__":
    main()
