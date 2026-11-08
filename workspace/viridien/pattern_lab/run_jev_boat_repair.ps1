$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\hssil\Downloads\03_Programming_Projects\hacktheburgh'
$taskOut = Join-Path $taskRoot 'viridien_work\pattern_lab\data\jev-e2e-20261108'
$taskOwner = Get-Content -LiteralPath (Join-Path $taskOut 'boat-control-owner.json') -Raw | ConvertFrom-Json
$taskActive = Get-CimInstance Win32_Process -Filter "ProcessId=$($taskOwner.process_id)"
if ($taskActive) {
    if ($taskActive.CreationDate.ToUniversalTime().Ticks -ne ([datetime]$taskOwner.creation_time).ToUniversalTime().Ticks -or $taskActive.CommandLine -notlike '*jev_boat_control.py run*') {
        throw 'The saved controller PID has a different owner. No repair was started.'
    }
    'Waiting for the first bounded control pass. No CPU lock is held by this wait.'
    Wait-Process -Id $taskActive.ProcessId -Timeout 15000 -ErrorAction Stop
}
'Starting one bounded repair pass. Completed case IDs are skipped.'
Set-Location -LiteralPath $taskRoot
& 'C:\Users\hssil\anaconda3\python.exe' -X utf8 -u viridien_work/pattern_lab/jev_boat_control.py run
$taskExit = $LASTEXITCODE
[ordered]@{state='finished';exit_code=$taskExit;passes=2;stop_rule='No further automatic pass';finished_utc=[datetime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskOut 'boat-repair-exit.json') -Encoding utf8
exit $taskExit
