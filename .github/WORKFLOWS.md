# GitHub Workflows Automation

This repository includes automated workflows for handling pull requests, approvals, conflict resolution, and merging.

## Workflows Overview

### 1. Auto-Approve and Merge (`auto-approve-and-merge.yml`)

**Triggers:** PR opened, synchronized, or reopened

**Purpose:** Automatically approve and merge PRs from trusted sources

**Features:**
- ✅ Filters PRs by author (bots: dependabot, renovate, github-actions, or copilot branches)
- ✅ Checks for merge conflicts before approval
- ✅ Waits for CI checks to pass before merging
- ✅ Uses squash merge strategy
- ✅ Comments on PRs with conflicts

**Permissions Required:**
- `contents: write` - To merge PRs
- `pull-requests: write` - To approve PRs

**Conditions for Auto-Approval:**
1. PR is from a bot or copilot branch
2. No merge conflicts exist
3. PR is not a draft

**Conditions for Auto-Merge:**
1. PR has been auto-approved
2. All CI checks pass
3. PR mergeable state is "clean"

---

### 2. Resolve PR Conflicts (`resolve-conflicts.yml`)

**Triggers:** 
- PR opened, synchronized, or reopened
- Daily schedule (midnight UTC)
- Manual dispatch

**Purpose:** Detect and automatically resolve merge conflicts

**Features:**
- 🔍 Detects merge conflicts in PRs
- 🔧 Attempts automatic resolution using merge strategies
- 🏷️ Adds/removes "merge-conflict" label
- 💬 Comments with instructions if manual resolution needed
- 📅 Runs daily to catch new conflicts

**Auto-Resolution Strategy:**
- Only auto-resolves specific file types (lock files)
- For lock files: package-lock.json, yarn.lock, Gemfile.lock, poetry.lock, Pipfile.lock
- Uses `--theirs` strategy only for these safe file types
- Other files require manual resolution
- Falls back to manual instructions if auto-resolution not possible

**When Manual Resolution is Needed:**
The workflow provides step-by-step instructions in a PR comment:
```bash
git checkout <branch>
git merge origin/<base-branch>
# Resolve conflicts manually
git commit -am "Resolve merge conflicts"
git push
```

---

### 3. Bot PR Handler (`bot-pr-handler.yml`)

**Triggers:** PR opened, synchronized, reopened, or labeled (for bot PRs only)

**Purpose:** Specialized handling for bot-submitted PRs

**Features:**
- 🤖 Specifically targets bot PRs (dependabot, renovate, github-actions)
- 🏷️ Automatically labels PRs: `bot-pr`, `auto-merge`, `dependencies`
- 💬 Adds welcome comment explaining automation process
- ✅ Checks PR readiness (not draft, mergeable)
- 🚀 Enables auto-merge when ready
- 📊 Provides status updates

**Process Flow:**
1. Bot opens PR
2. Workflow adds labels and welcome comment
3. Checks if PR is ready (not draft, mergeable)
4. Auto-approves if ready
5. Enables auto-merge
6. Comments on status

---

## Configuration

### Trusted Contributors

To add more trusted contributors for auto-approval, edit the `if` condition in `auto-approve-and-merge.yml`:

```yaml
if: |
  github.event.pull_request.user.login == 'dependabot[bot]' ||
  github.event.pull_request.user.login == 'renovate[bot]' ||
  github.event.pull_request.user.login == 'github-actions[bot]' ||
  github.event.pull_request.user.login == 'trusted-user' ||  # Add here
  contains(github.event.pull_request.head.ref, 'copilot/')
```

### Merge Strategy

The default merge strategy is `squash`. To change it, update the `merge-method` in the workflows:

```yaml
merge-method: squash  # Options: merge, squash, rebase
```

### Conflict Resolution Strategy

Conflict resolution uses `--strategy-option=theirs` by default. To customize:

Edit `resolve-conflicts.yml` and change the merge strategy:
```bash
git merge origin/${{ github.event.pull_request.base.ref }} --strategy-option=ours
```

---

## Security Considerations

1. **Event Types:**
   - All workflows use `pull_request` event type with filtered conditions
   - This ensures code runs in the context of the PR (safer for bot PRs)

2. **Permissions:**
   - Workflows use minimal required permissions
   - `GITHUB_TOKEN` is automatically provided and scoped

3. **Branch Protection:**
   - Consider enabling branch protection rules
   - Require status checks before merging
   - Require approvals for non-bot PRs

---

## Monitoring

### Check Workflow Status

1. Go to **Actions** tab in GitHub repository
2. View recent workflow runs
3. Click on individual runs for details

### Common Issues

**Issue:** PR not auto-approved
- **Solution:** Check if PR author matches trusted list
- **Solution:** Verify no merge conflicts exist

**Issue:** PR not auto-merging
- **Solution:** Check if all CI checks pass
- **Solution:** Verify PR is not a draft
- **Solution:** Check mergeable state

**Issue:** Conflict resolution failed
- **Solution:** Follow manual resolution instructions in PR comment
- **Solution:** Use `git mergetool` for complex conflicts

---

## Examples

### Example 1: Dependabot PR
1. Dependabot opens PR for dependency update
2. `bot-pr-handler.yml` adds labels and welcome comment
3. `resolve-conflicts.yml` checks for conflicts
4. `auto-approve-and-merge.yml` approves and enables auto-merge
5. PR merges automatically when CI passes

### Example 2: Copilot PR with Conflicts
1. Copilot creates PR with branch name `copilot/add-feature`
2. `resolve-conflicts.yml` detects conflicts
3. Workflow attempts auto-resolution
4. If successful, PR proceeds to approval
5. If not, comment added with manual instructions

### Example 3: Manual Workflow Trigger
1. Go to Actions → "Resolve PR Conflicts"
2. Click "Run workflow"
3. Select branch/PR to check
4. Workflow runs conflict check and resolution

---

## Customization

### Add Custom Labels

Edit `bot-pr-handler.yml`:
```javascript
const labels = ['bot-pr', 'auto-merge', 'custom-label'];
```

### Change Schedule

Edit `resolve-conflicts.yml` cron schedule:
```yaml
schedule:
  - cron: '0 */6 * * *'  # Every 6 hours
```

### Add Notifications

Add Slack/Discord notifications by adding steps:
```yaml
- name: Notify on Slack
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## Testing

### Test Workflows Locally

Use [act](https://github.com/nektos/act) to test workflows locally:

```bash
# Install act
brew install act  # macOS
# or
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

# Test a workflow
act pull_request -W .github/workflows/auto-approve-and-merge.yml
```

### Test with Draft PRs

1. Create a draft PR
2. Workflows should not auto-approve
3. Mark PR as ready for review
4. Workflows should proceed

---

## Support

For issues or questions:
1. Check workflow run logs in Actions tab
2. Review this documentation
3. Open an issue in the repository

---

## License

These workflows are part of the continual-learning repository and follow the same license.
