# Quick Reference: PR Automation Workflows

## What Gets Auto-Approved and Merged?

PRs from these sources are automatically processed:
- ✅ `dependabot[bot]` - Dependency updates
- ✅ `renovate[bot]` - Dependency management
- ✅ `github-actions[bot]` - GitHub Actions bot
- ✅ Branches starting with `copilot/` - GitHub Copilot PRs

## Workflow Process

### For Bot PRs:
1. **Bot opens PR** → `bot-pr-handler.yml` triggers
   - Adds labels: `bot-pr`, `auto-merge`, `dependencies` (if dependabot)
   - Adds welcome comment with automation status

2. **Conflict check** → `resolve-conflicts.yml` triggers
   - Checks for merge conflicts
   - Attempts automatic resolution
   - Adds instructions if manual resolution needed

3. **Auto-approval** → `auto-approve-and-merge.yml` triggers
   - Verifies no conflicts
   - Approves the PR
   - Waits for CI checks

4. **Auto-merge** → Merges when ready
   - All CI checks pass
   - PR is approved
   - No conflicts
   - Uses squash merge

## Manual Actions

### Manually Trigger Conflict Resolution:
1. Go to **Actions** tab
2. Select **"Resolve PR Conflicts"**
3. Click **"Run workflow"**
4. Select the branch

### Disable Auto-Merge for a PR:
Add label `no-auto-merge` to the PR (feature not yet implemented, but can be added)

### Check Workflow Status:
- Go to **Actions** tab
- View recent runs
- Check logs for details

## Common Scenarios

### Scenario 1: Dependabot PR with No Conflicts
- ✅ Auto-labeled
- ✅ Auto-approved
- ✅ Auto-merged (after CI passes)
- ⏱️ Time: ~2-5 minutes

### Scenario 2: PR with Merge Conflicts
- ⚠️ Conflict detected
- 🔧 Auto-resolution attempted
- ✅ If successful: proceeds to approval
- ❌ If failed: comment with manual instructions

### Scenario 3: Copilot PR
- ✅ Auto-approved (if no conflicts)
- ✅ Auto-merged (after CI passes)
- Uses branch name pattern: `copilot/*`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| PR not auto-approved | Check if author is in trusted list |
| PR not merging | Check CI status, conflicts, draft status |
| Conflict not resolved | Follow manual instructions in PR comment |
| Workflow not running | Check workflow file syntax, permissions |

## File Locations

- Main workflow: `.github/workflows/auto-approve-and-merge.yml`
- Conflict resolution: `.github/workflows/resolve-conflicts.yml`
- Bot handler: `.github/workflows/bot-pr-handler.yml`
- Full documentation: `.github/WORKFLOWS.md`

## Permissions Required

All workflows need:
```yaml
permissions:
  contents: write       # To merge PRs
  pull-requests: write  # To approve PRs
  issues: write         # To comment and label
```

## Customization

To add more trusted users, edit the `if` condition in each workflow:
```yaml
github.event.pull_request.user.login == 'your-trusted-user'
```

To change merge strategy:
```yaml
merge-method: squash  # Options: merge, squash, rebase
```
