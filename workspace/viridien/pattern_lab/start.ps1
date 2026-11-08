param([switch]$Restart,[switch]$NoBrowser)
$ErrorActionPreference='Stop'
$taskPort=8776
$taskPhysical=@(Get-NetAdapter -Physical | Select-Object -ExpandProperty ifIndex)
$taskAddress=Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceIndex -in $taskPhysical -and $_.IPAddress -notmatch '^127\.|^169\.254\.' -and $_.AddressState -eq 'Preferred' } | Sort-Object @{Expression={if($_.InterfaceAlias -match 'Wi-Fi|WLAN'){0}else{1}}} | Select-Object -First 1
$taskHost=if($taskAddress){$taskAddress.IPAddress}else{'127.0.0.1'}
$taskShareUrl="http://${taskHost}:$taskPort"
$taskUrl="http://127.0.0.1:$taskPort"
$taskRunning=$false
try {$response=Invoke-RestMethod "$taskUrl/api/health" -TimeoutSec 2;$taskRunning=($response.ok -and $response.documents -eq 51)} catch {$taskRunning=$false}
if($Restart -and $taskRunning){
    $taskPids=@(Get-NetTCPConnection -LocalPort $taskPort -State Listen | Select-Object -ExpandProperty OwningProcess -Unique)
    foreach($taskPid in $taskPids){
        $taskProcess=Get-CimInstance Win32_Process -Filter "ProcessId = $taskPid"
        if($taskProcess.CommandLine -notmatch 'server\.py' -or $taskProcess.ExecutablePath -notmatch 'python(?:w)?\.exe$'){throw 'The port belongs to an unknown process. No process was stopped.'}
        Stop-Process -Id $taskPid
    }
    $taskRunning=$false
}
if(-not $taskRunning){
    Start-Process -WindowStyle Hidden -FilePath python -ArgumentList '-X','utf8','server.py','--port',"$taskPort",'--host','127.0.0.1','--share-url',$taskShareUrl -WorkingDirectory $PSScriptRoot -RedirectStandardOutput "$PSScriptRoot/data/server.log" -RedirectStandardError "$PSScriptRoot/data/server-error.log"
    Write-Output "Server start sent. Open $taskUrl when it is ready."
}else{Write-Output "Shared review is running at $taskUrl"}
Write-Output "Reviewers can sync through $taskShareUrl while this computer is running."
if(-not $NoBrowser){Start-Process $taskUrl}
