$ErrorActionPreference='Stop'
$env:PLAYWRIGHT_MCP_EXECUTABLE_PATH='C:/Users/hssil/AppData/Local/ms-playwright/chromium-1248/chrome-win64/chrome.exe'
foreach($case in @('browser-auto-legend-check.js','browser-feedback-check.js','browser-shared-check.js')){
    $result=& npx --yes --package @playwright/cli playwright-cli -s=auto-legend --raw run-code --filename="viridien_work/pattern_lab/tests/$case" 2>&1
    $result | Write-Output
    if($LASTEXITCODE -ne 0 -or ($result -join "`n") -notmatch '"result"\s*:\s*"PASS"'){throw "Browser check failed: $case"}
}
