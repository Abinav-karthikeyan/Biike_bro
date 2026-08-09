---
name: auto-commit
description: Auto-commits code changes with generated one-liner messages during Claude Code work. Triggers when you say "commit", "commit these changes", "save progress", or when Claude completes a major task (substantial code changes, file edits, refactors). Use this to maintain a detailed commit history that auto-documents your work without manual message writing. Great for retrospection and understanding what changed and when.
compatibility: Git repository required
---

# Auto-Commit Skill

This skill automatically creates git commits with generated one-liner messages whenever:
1. **You explicitly request it** — phrases like "commit this", "commit my changes", "save progress", "commit"
2. **Claude completes a major task** — after substantial code changes, refactors, or file edits

The commit messages are concise, conventional-commit style one-liners that summarize what changed.

## How it works

When triggered, the skill:
1. Checks git status to see what files have changed
2. Stages relevant changes (excluding sensitive files like .env, secrets)
3. Generates a one-liner commit message based on the diff
4. Creates the commit
5. Reports the commit hash and message for your log

## Commit message format

Messages follow conventional commit style for clarity:
- `feat:` New feature or capability
- `fix:` Bug fix
- `refactor:` Code restructuring without behavior change
- `docs:` Documentation updates
- `style:` Formatting, whitespace, linting
- `test:` Test additions or modifications
- `chore:` Dependency updates, tooling

**Examples:**
```
feat: add HNSW vector search integration
fix: correct zone prediction accuracy calculation
refactor: simplify data loading pipeline
docs: update API endpoint documentation
```

## Triggering the skill

### Explicit trigger (user request)
Say any of:
- "commit this"
- "commit my changes"
- "commit the changes"
- "save progress"
- "git commit"
- "make a commit"

### Automatic trigger (task completion)
After Claude completes work like:
- Multiple file edits addressing a task
- Significant refactoring
- New feature implementation
- Bug fixes with multiple files changed

The skill checks if enough meaningful changes exist to warrant a commit, and asks before proceeding.

## What gets committed

The skill stages and commits:
- Source code changes (.py, .ts, .js, .go, .rs, etc.)
- Configuration files (config.json, .yaml, .toml)
- Documentation (*.md, docs/**)

The skill **excludes by default**:
- `.env` and credential files
- `node_modules`, `__pycache__`, build artifacts
- Temporary/scratch files
- Large binary files (unless explicitly included)

## Safety features

- **Confirmation step** — the skill shows you what will be committed before creating the commit
- **Readable diffs** — you see a summary of changed lines per file
- **Dry-run available** — you can preview the commit message without committing
- **Skip option** — you can skip the auto-commit if you prefer manual control for a specific change

## Retrospection

After work is done, review your commit history:
```bash
git log --oneline -20
# or with diffs
git log -p --oneline -5
```

Each one-liner message becomes part of your project history, making it easy to scan what was done when.

## Example workflow

```
You: "Add validation to the user signup form"
Claude: [Makes changes to form.ts, validation.ts, tests.ts]
Skill: "Ready to commit? Changed 3 files with 145 lines added/modified. Message: feat: add email validation to signup form"
You: "Commit"
Skill: ✓ Committed as abc1234 (feat: add email validation to signup form)

You: "Now let's fix that bug in the parser"
Claude: [Edits parser.py, adds test]
Skill: [Auto-triggers after major change] "Detected substantial changes. Committing as fix: handle edge case in parser error recovery"
Skill: ✓ Committed as def5678 (fix: handle edge case in parser error recovery)

[Later]
You: git log --oneline -10
# Shows all your changes neatly documented
```

## If something goes wrong

If the commit message feels wrong:
1. The commit is already made, but you can amend it: `git commit --amend -m "better message"`
2. Or revert: `git revert [commit-hash]`
3. Or use interactive rebase to clean up: `git rebase -i HEAD~5`

The skill errs on the side of committing (you asked for it or the change is substantial) rather than refusing to commit.
