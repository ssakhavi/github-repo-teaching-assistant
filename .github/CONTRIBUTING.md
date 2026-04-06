# Contributing

## Receiving upstream updates

This repo was created from the [github-repo-teaching-assistant](https://github.com/ssakhavi/github-repo-teaching-assistant) template.

The `sync-upstream` workflow runs every Monday and opens a pull request whenever the template's infrastructure files are updated:

- `src/` — terminal interface
- `.claude/commands/` — slash command definitions
- `CLAUDE.md` — Claude Code project instructions
- `pyproject.toml` / `uv.lock` — dependencies
- `README.md` / `.gitignore`

**Your training data, input files, and personal customisations are never touched.**

To trigger a sync manually: **Actions → Sync from upstream template → Run workflow.**

## Contributing back to the template

To suggest improvements to the template itself, open a pull request against [ssakhavi/github-repo-teaching-assistant](https://github.com/ssakhavi/github-repo-teaching-assistant).
