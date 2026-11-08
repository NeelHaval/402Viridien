param([switch]$Restart)
$ErrorActionPreference='Stop'
$workspace=Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..')
$running=@(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '-m\s+genlab\.assemble\.stats\s+--watch'
})
if($Restart){
    foreach($process in $running){Stop-Process -Id $process.ProcessId -ErrorAction SilentlyContinue}
    $running=@()
}
if($running.Count){Write-Output "PASS stats watcher alive pid=$($running[0].ProcessId)";exit 0}
$python=(Get-Command python).Source
$log=Join-Path $PSScriptRoot '..\stats-watch.log'
$command="cmd /c `"`"$python`" -X utf8 -m genlab.assemble.stats --watch >> `"$log`" 2>&1`""
$result=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine=$command;CurrentDirectory=$workspace.Path
}
if($result.ReturnValue -ne 0){throw "WMI create failed: $($result.ReturnValue)"}
Write-Output "PASS stats watcher detached via WMI parent=$($result.ProcessId)"
