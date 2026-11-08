$ErrorActionPreference = 'Stop'
$taskPid = 57456
$taskCreated = '2026-11-08T06:41:27.614201+00:00'
$taskRoot = 'C:\Users\hssil\Downloads\03_Programming_Projects\hacktheburgh\viridien_work\pattern_lab\data\jev-e2e-20261108'
$taskPath = Join-Path $taskRoot 'results.json'
$taskInitial = (Get-Content -LiteralPath $taskPath -Raw | ConvertFrom-Json).records.Count
$taskWatcher = [System.IO.FileSystemWatcher]::new($taskRoot, 'results.json')
$taskWatcher.EnableRaisingEvents = $true
$taskEvent = $taskWatcher.WaitForChanged([System.IO.WatcherChangeTypes]::All, 600000)
$taskWatcher.Dispose()
if ($taskEvent.TimedOut) { throw 'No new Jev checkpoint within 600 seconds; worker was not stopped.' }
$taskSaved = Get-Content -LiteralPath $taskPath -Raw | ConvertFrom-Json
if ($taskSaved.records.Count -le $taskInitial) { throw 'Checkpoint count did not advance; worker was not stopped.' }
$taskWorker = Get-CimInstance Win32_Process -Filter "ProcessId = $taskPid"
if ($null -eq $taskWorker) { throw 'Worker already exited.' }
if ($taskWorker.CreationDate.ToUniversalTime().ToString('o') -ne ([datetime]::Parse($taskCreated)).ToUniversalTime().ToString('o')) { throw 'Worker creation time changed.' }
if ($taskWorker.CommandLine -notlike '*python.exe* -X utf8 viridien_work/pattern_lab/jev_e2e.py run') { throw 'Worker command does not match.' }
$taskHasher = [System.Security.Cryptography.SHA256]::Create()
$taskHash = ([System.BitConverter]::ToString($taskHasher.ComputeHash([System.IO.File]::ReadAllBytes($taskPath)))).Replace('-', '').ToLowerInvariant()
$taskHasher.Dispose()
Stop-Process -Id $taskPid
$taskReceipt = @{ process_id=$taskPid; creation_time=$taskCreated; command=$taskWorker.CommandLine; checkpoint_cases=$taskSaved.records.Count; checkpoint_sha256=$taskHash; last_case=$taskSaved.records[-1].case_id; reason='Owner requested VM offload. Stopped this exact worker only, immediately after its next atomic case checkpoint.' }
$taskReceipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $taskRoot 'local-worker-release.json') -Encoding UTF8
$taskReceipt | ConvertTo-Json -Compress
