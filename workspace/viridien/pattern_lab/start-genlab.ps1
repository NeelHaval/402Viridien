param([switch]$Restart,[switch]$NoBrowser)
# Starts the pattern generator lab on http://127.0.0.1:8781.
# It watches data/reviews and rebuilds whenever a box is accepted in the labeller.
$ErrorActionPreference='Stop'
$port=8781
$url="http://127.0.0.1:$port"
$running=$false
try {Invoke-RestMethod "$url/state.json" -TimeoutSec 2 | Out-Null; $running=$true} catch {$running=$false}
if($Restart -and $running){
    foreach($p in @(Get-NetTCPConnection -LocalPort $port -State Listen | Select-Object -ExpandProperty OwningProcess -Unique)){
        $proc=Get-CimInstance Win32_Process -Filter "ProcessId = $p"
        if($proc.CommandLine -notmatch 'genlab\.server'){throw 'Port 8781 belongs to another program. Nothing was stopped.'}
        Stop-Process -Id $p
    }
    $running=$false
}
if(-not $running){
    # Created through WMI so the server outlives the shell that started it.
    $python=(Get-Command python).Source
    $log="$PSScriptRoot\data\genlab-server.log"
    $cmd="cmd /c `"`"$python`" -X utf8 -m genlab.server --port $port > `"$log`" 2>&1`""
    $r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd; CurrentDirectory="$PSScriptRoot"}
    if($r.ReturnValue -ne 0){throw "Could not start the server (WMI code $($r.ReturnValue))."}
    Write-Output "Generator lab starting at $url (first build takes about 40 s)."
}else{Write-Output "Generator lab is running at $url"}
if(-not $NoBrowser){Start-Process $url}
