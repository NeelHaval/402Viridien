param([switch]$Restart)
# Kept inside genlab because the task permits source edits only there.
$ErrorActionPreference='Stop'
$workspace=Split-Path -Parent $PSScriptRoot
$output=Join-Path $workspace 'data\genlab\datasets'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$running=@(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '-m\s+genlab\.dataset\s+loop(?:\s|$)'
})
if($Restart){
    foreach($process in $running){Stop-Process -Id $process.ProcessId -ErrorAction SilentlyContinue}
    $running=@()
}
if($running.Count -gt 0){
    Write-Output "PASS dataset loop already running pid=$($running[0].ProcessId)"
    exit 0
}
$python=(Get-Command python).Source
$log=Join-Path $output 'loop.log'
$command="cmd /c `"`"$python`" -X utf8 -u -m genlab.dataset loop >> `"$log`" 2>&1`""
$result=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine=$command; CurrentDirectory=$workspace
}
if($result.ReturnValue -ne 0){throw "WMI create failed: $($result.ReturnValue)"}
Write-Output "PASS detached dataset loop launched via WMI parent=$($result.ProcessId); log=$log"
