#!/usr/bin/env python3
"""
Auto-commit script: Stage changes and create git commits with generated messages.
"""

import subprocess
import json
import sys
import os
from pathlib import Path
from typing import Optional, Tuple

# Files to exclude from commits
EXCLUDE_PATTERNS = {
    '.env', '.env.local', '.env.*.local',
    'credentials.json', 'secrets.json', 'token.json',
    'node_modules/', '__pycache__/', '.venv/', 'venv/',
    '.pytest_cache/', '.coverage', 'dist/', 'build/',
    '*.pyc', '*.pyo', '*.egg-info',
    '.DS_Store', 'Thumbs.db'
}

INCLUDE_EXTENSIONS = {
    '.py', '.ts', '.tsx', '.js', '.jsx', '.go', '.rs', '.java', '.cpp', '.c', '.h',
    '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.md', '.rst', '.txt', '.sh', '.bash',
    '.html', '.css', '.scss', '.vue',
}

def run_cmd(cmd: str, check: bool = True) -> str:
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running: {cmd}", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        return ""

def get_git_status() -> dict:
    """Get current git status."""
    output = run_cmd("git status --porcelain")
    if not output:
        return {"modified": [], "untracked": [], "deleted": []}

    files = {"modified": [], "untracked": [], "deleted": []}
    for line in output.split('\n'):
        if not line:
            continue
        status = line[:2]
        filepath = line[3:]

        if status == ' M':  # Modified
            files["modified"].append(filepath)
        elif status == '??':  # Untracked
            files["untracked"].append(filepath)
        elif status == ' D':  # Deleted
            files["deleted"].append(filepath)

    return files

def should_include_file(filepath: str) -> bool:
    """Check if file should be included in commit."""
    # Exclude if matches exclude patterns
    for pattern in EXCLUDE_PATTERNS:
        if pattern.endswith('/'):
            if filepath.startswith(pattern):
                return False
        else:
            if filepath == pattern or filepath.endswith(pattern):
                return False

    # Include if has relevant extension
    ext = Path(filepath).suffix.lower()
    if ext in INCLUDE_EXTENSIONS:
        return True

    # Include certain config files by name
    name = Path(filepath).name.lower()
    if name in ['dockerfile', 'makefile', 'procfile', '.gitignore', '.gitattributes']:
        return True

    return False

def get_diff_summary(files: dict) -> str:
    """Generate a summary of what changed."""
    summary = []

    if files["modified"]:
        summary.append(f"Modified: {len(files['modified'])} file(s)")
    if files["untracked"]:
        summary.append(f"Created: {len(files['untracked'])} file(s)")
    if files["deleted"]:
        summary.append(f"Deleted: {len(files['deleted'])} file(s)")

    return ", ".join(summary)

def stage_files(files: dict) -> list:
    """Stage relevant files for commit."""
    to_stage = []

    for filepath in files["modified"] + files["untracked"]:
        if should_include_file(filepath):
            to_stage.append(filepath)

    for filepath in files["deleted"]:
        if should_include_file(filepath):
            to_stage.append(filepath)

    if not to_stage:
        return []

    # Stage files
    for filepath in to_stage:
        run_cmd(f'git add "{filepath}"')

    return to_stage

def get_commit_type_and_scope(staged_files: list, changes_summary: str) -> Tuple[str, str]:
    """Infer commit type and scope from staged files."""

    # Simple heuristics based on file patterns and count
    has_tests = any('test' in f or 'spec' in f for f in staged_files)
    has_docs = any('.md' in f or 'docs' in f for f in staged_files)
    has_config = any(f.endswith(('.json', '.yaml', '.yml', '.toml', '.cfg')) for f in staged_files)
    has_source = any(f.endswith(('.py', '.ts', '.tsx', '.js', '.jsx', '.go', '.rs')) for f in staged_files)

    file_count = len(staged_files)

    # Determine type
    if file_count == 1:
        # Single file - look at content
        filepath = staged_files[0]
        if 'test' in filepath or 'spec' in filepath:
            commit_type = "test"
        elif '.md' in filepath or 'docs' in filepath:
            commit_type = "docs"
        elif has_config:
            commit_type = "chore"
        else:
            # Default to feat or fix - will need more context
            # Check for keywords in diff
            diff = run_cmd(f'git diff --staged "{filepath}"')
            if 'fix' in diff.lower() or 'bug' in diff.lower() or 'error' in diff.lower():
                commit_type = "fix"
            else:
                commit_type = "feat"
    else:
        # Multiple files - use heuristics
        if has_source and not has_tests and not has_docs:
            commit_type = "feat"  # Assume feature unless it looks like a fix
        elif has_tests and not has_source:
            commit_type = "test"
        elif has_docs and not has_source:
            commit_type = "docs"
        elif file_count > 5:
            commit_type = "refactor"  # Lots of files likely refactor
        else:
            commit_type = "feat"

    # Determine scope from primary file/directory
    if staged_files:
        main_file = staged_files[0]
        parts = main_file.split('/')
        # Try to use directory or file stem as scope
        if len(parts) > 1:
            scope = parts[0]  # Use top-level directory
        else:
            scope = Path(main_file).stem  # Use filename without extension

        scope = scope.replace('_', '-').replace(' ', '-').lower()[:30]  # Clean up
    else:
        scope = "changes"

    return commit_type, scope

def generate_commit_message(staged_files: list, changes_summary: str) -> str:
    """Generate a one-liner commit message."""

    if not staged_files:
        return "chore: minor updates"

    commit_type, scope = get_commit_type_and_scope(staged_files, changes_summary)

    # Read diff to extract changes
    diff_output = run_cmd("git diff --staged --stat")

    # Try to infer what changed
    keywords = []
    diff = run_cmd("git diff --staged")

    # Look for common patterns
    if any(word in diff.lower() for word in ['add', 'implement', 'implement', 'new']):
        keywords.append("add")
    if any(word in diff.lower() for word in ['fix', 'correct', 'resolve', 'bug']):
        keywords.append("fix")
    if any(word in diff.lower() for word in ['refactor', 'simplify', 'restructure']):
        keywords.append("refactor")
    if any(word in diff.lower() for word in ['update', 'improve', 'enhance']):
        keywords.append("update")

    # Build message
    if commit_type == "fix":
        if len(staged_files) == 1:
            detail = f"in {Path(staged_files[0]).stem}"
        else:
            detail = f"{len(staged_files)} files"
        message = f"fix: correct {detail}"
    elif commit_type == "test":
        message = f"test: add tests for {scope}"
    elif commit_type == "docs":
        message = f"docs: update documentation"
    elif commit_type == "refactor":
        message = f"refactor: improve {scope} code structure"
    else:  # feat or default
        if len(staged_files) == 1:
            detail = f"to {Path(staged_files[0]).stem}"
        else:
            detail = f"({len(staged_files)} files)"
        message = f"feat: implement changes {detail}"

    return message[:72]  # Keep under 72 chars for conventional commit

def confirm_commit(staged_files: list, message: str) -> bool:
    """Ask user to confirm the commit."""
    print("\n" + "="*60)
    print("READY TO COMMIT")
    print("="*60)
    print(f"\nCommit message: {message}")
    print(f"\nFiles to commit ({len(staged_files)}):")
    for f in sorted(staged_files)[:10]:
        print(f"  - {f}")
    if len(staged_files) > 10:
        print(f"  ... and {len(staged_files) - 10} more files")

    print("\nOptions: [c]ommit, [a]mend message, [s]kip, [d]ry-run")
    response = input("\nProceed? (c/a/s/d): ").strip().lower()

    if response == 'a':
        new_msg = input("New message: ").strip()
        if new_msg:
            return confirm_commit(staged_files, new_msg)
        return confirm_commit(staged_files, message)
    elif response == 'd':
        print("\n[DRY RUN] Would commit with message:")
        print(f"  {message}")
        return False
    elif response == 's':
        return False

    return response != 's'

def create_commit(message: str) -> Optional[str]:
    """Create the git commit."""
    try:
        # Use Co-Authored-By to note this was auto-generated
        commit_msg = f"{message}\n\nCo-Authored-By: Claude Auto-Commit <auto-commit@claude>"
        run_cmd(f'git commit -m "{commit_msg}"')

        # Get the commit hash
        commit_hash = run_cmd("git rev-parse HEAD")[:7]
        return commit_hash
    except Exception as e:
        print(f"Failed to create commit: {e}", file=sys.stderr)
        return None

def main():
    """Main entry point."""

    # Check if in git repo
    if run_cmd("git rev-parse --git-dir", check=False) == "":
        print("Error: Not in a git repository", file=sys.stderr)
        sys.exit(1)

    # Get git status
    status = get_git_status()

    if not status["modified"] and not status["untracked"] and not status["deleted"]:
        print("✓ No changes to commit")
        sys.exit(0)

    # Summary
    summary = get_diff_summary(status)
    print(f"Changes detected: {summary}")

    # Stage files
    staged = stage_files(status)
    if not staged:
        print("✓ No files matched commit criteria (likely sensitive files excluded)")
        sys.exit(0)

    # Generate message
    message = generate_commit_message(staged, summary)

    # Confirm and commit
    if confirm_commit(staged, message):
        commit_hash = create_commit(message)
        if commit_hash:
            print(f"\n✓ Committed: {commit_hash}")
            print(f"  Message: {message}")
        else:
            print("✗ Failed to create commit")
            sys.exit(1)
    else:
        print("✓ Commit skipped")
        # Unstage files
        run_cmd("git reset")

if __name__ == "__main__":
    main()
