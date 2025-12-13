# Session Logs

Audit trail and context reference for Claude Code sessions.

## Purpose

- Track significant development sessions for audit compliance
- Provide context for future sessions (continuation/reference)
- Document architectural decisions and rationale
- Record completed tasks and outcomes

## File Naming Convention

```
YYYYMMDD_HHMMSS_{slug}.md
```

Examples:
- `20251212_143022_refactoring-phase1.md`
- `20251212_180500_bugfix-unicode-paths.md`

## Log Retention

- **Active logs**: Current sprint/month
- **Archive**: Move older logs to `archive/` subdirectory
- **Sensitive data**: Never include credentials, API keys, or PII

## Usage

1. Copy `TEMPLATE.md` to create new log
2. Fill in session metadata
3. Record key decisions, changes, and outcomes
4. Commit with related code changes (or separately for audit-only)
