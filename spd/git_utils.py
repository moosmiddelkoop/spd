"""Git utilities for creating code snapshots."""

import datetime
import subprocess
import tempfile
from pathlib import Path

from spd.settings import REPO_ROOT


def create_git_snapshot(branch_name_prefix: str) -> str:
    """Create a git snapshot branch with current changes and push to origin.

    Creates a timestamped branch containing all current changes (staged and unstaged).
    Uses a temporary worktree to avoid affecting the current working directory.
    Pushes the snapshot branch to origin so other users can access it.

    Returns:
        Branch name of the created snapshot

    Raises:
        subprocess.CalledProcessError: If git commands fail
    """
    # Generate timestamped branch name
    timestamp_utc = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    snapshot_branch = f"{branch_name_prefix}-{timestamp_utc}"

    # Create temporary worktree path
    with tempfile.TemporaryDirectory() as temp_dir:
        worktree_path = Path(temp_dir) / f"spd-snapshot-{timestamp_utc}"

        try:
            # Create worktree with new branch
            subprocess.run(
                ["git", "worktree", "add", "-b", snapshot_branch, str(worktree_path)],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
            )

            # Copy current working tree to worktree (including untracked files)
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--delete",
                    "--exclude=.git",
                    "--filter=:- .gitignore",
                    f"{REPO_ROOT}/",
                    f"{worktree_path}/",
                ],
                check=True,
                capture_output=True,
            )

            # Stage all changes in the worktree
            subprocess.run(["git", "add", "-A"], cwd=worktree_path, check=True, capture_output=True)

            # Check if there are changes to commit
            result = subprocess.run(
                ["git", "diff", "--cached", "--quiet"], cwd=worktree_path, capture_output=True
            )

            # Commit changes if any exist
            if result.returncode != 0:  # Non-zero means there are changes
                subprocess.run(
                    ["git", "commit", "-m", f"Sweep snapshot {timestamp_utc}", "--no-verify"],
                    cwd=worktree_path,
                    check=True,
                    capture_output=True,
                )
            print(f"Created git snapshot branch: {snapshot_branch}")

            subprocess.run(
                ["git", "push", "-u", "origin", snapshot_branch],
                cwd=worktree_path,
                check=True,
                capture_output=True,
            )

        finally:
            # Clean up worktree (branch remains in main repo)
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_path)],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
            )

    return snapshot_branch
