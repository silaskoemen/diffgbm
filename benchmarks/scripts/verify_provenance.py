"""Verify that every reported result row came from committed model source.

Each row written by the harness records the repository commit (`git_sha`), a
`git_dirty` flag, and a content hash of the model package
(`treeffuser_source_hash`; see `benchmarks/harness.py::_hash_diffgbm_source`).

`git_dirty` is `git status` over the *whole* repository, and the harness writes
its artifacts into tracked paths under `benchmarks/results/` while a run is in
progress, so a run that produces committed artifacts reports a dirty tree almost
by construction. That flag therefore says little about the model code.

The content hash is the claim worth checking: for every distinct
(commit, source hash) pair in the artifacts, recompute the hash from the
committed `src/` tree at that commit and confirm it matches. If it does, the row
was produced by model code byte-identical to a committed state, and no
uncommitted diff needs to be reconstructed to know what ran.

Usage:
    pixi run python -m benchmarks.scripts.verify_provenance
    pixi run python -m benchmarks.scripts.verify_provenance --results-dir <dir>

Exits non-zero if any pair fails to reproduce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks/results/tuning/eval"

# The package was renamed treeffuser -> diffgbm after the reported runs, so a
# historical commit may carry either directory name.
PACKAGE_NAMES = ("diffgbm", "treeffuser")


def _git(args: list[str]) -> str:
    completed = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def _hash_source_at_commit(commit: str, package: str) -> str | None:
    """Recompute `_hash_diffgbm_source` against a committed tree.

    Mirrors the harness byte-for-byte: walk the package's `.py` files in sorted
    order, feeding the package-relative path and then the file contents.
    """
    try:
        listing = _git(["ls-tree", "-r", "--name-only", commit, f"src/{package}/"])
    except subprocess.CalledProcessError:
        return None
    paths = sorted(p for p in listing.split() if p.endswith(".py"))
    if not paths:
        return None
    prefix = f"src/{package}/"
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path[len(prefix) :].encode("utf-8"))
        digest.update(
            subprocess.run(  # noqa: S603
                ["git", "show", f"{commit}:{path}"],  # noqa: S607
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
            ).stdout
        )
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    files = sorted(args.results_dir.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl artifacts found under {args.results_dir}", file=sys.stderr)
        return 2

    pairs: Counter[tuple[str, str]] = Counter()
    dirty = 0
    total = 0
    for path in files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            dirty += bool(row.get("git_dirty"))
            pairs[(row["git_sha"], row["treeffuser_source_hash"])] += 1

    print(f"{total} rows across {len(files)} artifact files")
    print(f"git_dirty=True on {dirty}/{total} rows (expected: the harness writes tracked artifacts mid-run)")
    print(f"{len(pairs)} distinct (commit, source-hash) pairs\n")

    failures = 0
    for (sha, source_hash), count in sorted(pairs.items(), key=lambda kv: -kv[1]):
        recomputed = next(
            (h for pkg in PACKAGE_NAMES if (h := _hash_source_at_commit(sha, pkg)) is not None),
            None,
        )
        ok = recomputed == source_hash
        failures += not ok
        status = "OK  " if ok else "FAIL"
        print(f"{status} {sha[:12]}  {source_hash[:16]}  {count:4d} rows")
        if not ok:
            print(f"     recomputed from committed tree: {recomputed}")

    print()
    if failures:
        print(f"{failures} pair(s) did NOT reproduce from committed source.")
        return 1
    print("All rows were produced by model source byte-identical to a committed state.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
