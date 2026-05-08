<#
.SYNOPSIS
    Register or unregister a Windows scheduled task that runs `drift run`.

.PARAMETER IntervalHours
    Interval in hours between runs. Default 6.

.PARAMETER Unregister
    If set, removes the task instead of creating it.

.PARAMETER TaskName
    Name of the scheduled task. Default "LLM-DriftMonitor".

.EXAMPLE
    .\scripts\task_scheduler.ps1
    Registers the task to run every 6 hours.

.EXAMPLE
    .\scripts\task_scheduler.ps1 -IntervalHours 12

.EXAMPLE
    .\scripts\task_scheduler.ps1 -Unregister
#>
[CmdletBinding()]
param(
    [int]$IntervalHours = 6,
    [switch]$Unregister,
    [string]$TaskName = "LLM-DriftMonitor"
)

$ErrorActionPreference = 'Stop'

if ($Unregister) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Output "Unregistered task '$TaskName'."
    } else {
        Write-Output "No task named '$TaskName' found."
    }
    exit 0
}

if ($IntervalHours -lt 1) {
    Write-Error "IntervalHours must be >= 1."
    exit 1
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$LogFile = Join-Path $LogDir "drift-run.log"

# Run via cmd.exe so we can append stdout+stderr to the log file in one line.
$Action = New-ScheduledTaskAction `
    -Execute "cmd.exe" `
    -Argument "/c uv run drift run >> `"$LogFile`" 2>&1" `
    -WorkingDirectory $RepoRoot

# First fire 2 minutes from now, then every $IntervalHours after that.
$Trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Hours $IntervalHours)

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "LLM Drift Detector — runs ``uv run drift run`` every $IntervalHours hour(s); logs to $LogFile" `
    -Force | Out-Null

Write-Output "Registered '$TaskName' to run every $IntervalHours hour(s)."
Write-Output "Logs:  $LogFile"
Write-Output "Inspect: Get-ScheduledTask -TaskName '$TaskName'"
Write-Output "Remove:  .\scripts\task_scheduler.ps1 -Unregister"
