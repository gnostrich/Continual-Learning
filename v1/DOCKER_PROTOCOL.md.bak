# CRITICAL: Docker-First Development Protocol

**EVERY command that executes code, installs packages, or modifies files MUST use Docker.**

## Automatic Redirects (Agent MUST do without asking)

Before executing ANY command, check:
- Is this Python/Node/etc execution? → `docker exec <container> <command>`
- Is this package install? → `docker exec <container> pip/npm install`
- Is this running tests? → `docker exec <container> pytest/npm test`
- Is this file modification? → Modify inside Docker volume mount

## Examples

❌ NEVER:
```bash
python script.py
pip install package
npm test
cat /Users/rohan/workspace/file.py
```

✅ ALWAYS:
```bash
docker exec <container> python script.py
docker exec <container> pip install package
docker exec <container> npm test
docker exec <container> cat /workspace/file.py
```

## Only Ask Approval For
- `open <file>` (host OS viewer)
- `git commit/push` (git operations)
- `gh` commands (GitHub CLI)

## Active Container for This Workspace
Check `docker ps` to find the active container name, or create one if needed.
