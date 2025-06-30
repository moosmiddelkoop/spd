#!/usr/bin/env python3
"""
migrate_checkpoints.py

Flatten & rename checkpoint directories.

    OLD: <root>/<run_name_part0>/<run_name_part1>_<timestamp>
    NEW: <root>/<timestamp>_<run_name_part0>_<run_name_part1>

You can **move** (default) or **copy** the directories.

Usage
-----
# Dry-run (just show the plan)
python migrate_checkpoints.py --root /path/to/checkpoints

# Actually move
python migrate_checkpoints.py --root /path/to/checkpoints --execute

# Copy instead of move
python migrate_checkpoints.py --root /path/to/checkpoints --execute --mode copy
"""
import argparse
import os
import re
import shutil
from pathlib import Path
from tqdm import tqdm

TIMESTAMP_RE = re.compile(
    r"""^(.+?)_               # run-name tail
        (\d{8})_              # YYYYMMDD
        (\d{6})_              # HHMMSS
        (\d{3})$              # mmm
    """,
    re.VERBOSE,
)

def build_new_name(rel_parts: list[str]) -> str | None:
    leaf = rel_parts[-1]
    m = TIMESTAMP_RE.match(leaf)
    if not m:
        return None

    run_tail, ymd, hms, msec = m.groups()
    timestamp = f"{ymd}_{hms}_{msec}"

    full_run_raw = "/".join(rel_parts[:-1] + [run_tail])
    sanitized_run = full_run_raw.replace("/", "_")

    return f"{timestamp}_{sanitized_run}"

def migrate(root: Path, execute: bool = False, mode: str = "move") -> None:
    assert mode in {"move", "copy"}, "mode must be 'move' or 'copy'"

    moves: list[tuple[Path, Path]] = []
    for dir_path, _, _ in os.walk(root, topdown=False):
        dir_path = Path(dir_path)
        rel_parts = dir_path.relative_to(root).parts
        if not rel_parts:          # skip the root itself
            continue

        new_name = build_new_name(list(rel_parts))
        if new_name:
            moves.append((dir_path, root / new_name))

    if not moves:
        print("Nothing to migrate.")
        return

    print(f"Planned migrations ({'dry-run' if not execute else mode.upper()}):")
    for src, dst in moves:
        print(f"  {src}  →  {dst}")

    if not execute:
        print("\nRun again with --execute to perform the operation.")
        return

    # Do the work
    for src, dst in tqdm(moves):
        if dst.exists():
            tqdm.write(f'warning: destination exists: {dst}')
            continue
            # print()
            # shutil.rmtree(dst)
        # if dst.exists():
        #     raise FileExistsError(f"Destination exists: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if mode == "move":
            shutil.move(src, dst)
        else:  # copy
            shutil.copytree(src, dst, dirs_exist_ok=False)

    # Clean up empties when moving
    if mode == "move":
        for dir_path, dir_names, file_names in os.walk(root, topdown=False):
            if dir_path == str(root):
                continue
            if not dir_names and not file_names:
                Path(dir_path).rmdir()

    print("\nMigration complete.")

if __name__ == "__main__":
    migrate(
        Path("spd/experiments/lm/out"),
        execute=True,
        mode="move"
    )
