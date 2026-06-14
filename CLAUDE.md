# CLAUDE.md

Working rules for Claude Code in this repository. Keep this file short and rule-focused.

## `info/` is off-limits

`info/` holds internal commercial data — pricing, BOM, final pricing, real product data, and product photos — that the maintainer edits by hand. It is gitignored.

- **Never create, edit, move, or delete anything under `info/` on your own initiative.**
- Touch a file in `info/` **only** when given an explicit instruction that names what to change there.
- Never stage, commit, or copy `info/` contents into tracked files unless explicitly told to.

## Keep everything else clean

Outside `info/`, when you come across code, files, config, or docs that are clearly outdated, dead, or unused, **remove them promptly** — don't leave them commented-out or orphaned, and don't let cruft accumulate.

- Verify it's truly unused first: grep for references, including dynamic/indirect ones, before deleting.
- When a change makes something obsolete, delete the obsolete thing as part of the same change.
