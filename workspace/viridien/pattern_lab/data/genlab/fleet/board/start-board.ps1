param([switch]$Restart)
$ErrorActionPreference = 'Stop'
$port = 8782
$url = "http://127.0.0.1:$port"
$root = (Resolve-Path (Join-Path $PSScriptRoot '../../../..')).Path
$script = Join-Path $PSScriptRoot 'board.py'
$listeners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
if ($Restart -and $listeners.Count) {
    foreach ($owner in @($listeners | Select-Object -ExpandProperty OwningProcess -Unique)) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $owner"
        if ($proc.CommandLine -notlike "*$script*") {
            throw 'Port 8782 belongs to another program. Nothing was stopped.'
        }
        Stop-Process -Id $owner
    }
    $listeners = @()
}
if (-not $listeners.Count) {
    $python = (Get-Command python).Source
    $log = Join-Path $PSScriptRoot 'board.log'
    $cmd = "cmd /c `"`"$python`" -B -X utf8 `"$script`" > `"$log`" 2>&1`""
    $result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd; CurrentDirectory=$root}
    if ($result.ReturnValue -ne 0) { throw "Could not start board (WMI code $($result.ReturnValue))." }
}
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $state = Invoke-RestMethod "$url/board.json" -TimeoutSec 3
        if ($null -ne $state.total) { Write-Output "Pattern fleet board: $url ($($state.total) legend entries)"; exit 0 }
    } catch { Start-Sleep -Milliseconds 500 }
}
throw "Board did not become ready. See $PSScriptRoot/board.log"
