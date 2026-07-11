param(
    [string]$LiveConfig = "configs/live_router.local.json",
    [string]$LaunchLock = "configs/demo_month_launch_lock.local.json",
    [string]$OpsConfig = "configs/live_ops.local.json"
)

$ErrorActionPreference = "Stop"

Write-Host "DEBCO demo-month server runner"
Write-Host "repo: $(Get-Location)"
Write-Host "python: $(python --version)"
Write-Host "git: $(git rev-parse --short HEAD)"

python scripts/20_run_demo_month_supervisor.py `
  --live-config $LiveConfig `
  --launch-lock $LaunchLock `
  --ops-config $OpsConfig
