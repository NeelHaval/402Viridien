$ErrorActionPreference='Stop'
Get-Command npx | Select-Object -ExpandProperty Name
$env:PLAYWRIGHT_MCP_EXECUTABLE_PATH='C:/Users/hssil/AppData/Local/ms-playwright/chromium-1248/chrome-win64/chrome.exe'
New-Item -ItemType Directory -Path output/playwright/terra-grid -Force | Out-Null
& npx --yes --package @playwright/cli playwright-cli -s=terra-grid open http://127.0.0.1:8777/review
if($LASTEXITCODE -ne 0){throw 'Could not open the QA browser.'}
$result=& npx --yes --package @playwright/cli playwright-cli -s=terra-grid --raw run-code --filename=viridien_work/pattern_lab/tests/browser-terra-grid-check.js 2>&1
$result | Write-Output
if($LASTEXITCODE -ne 0 -or ($result -join "`n") -notmatch '"result"\s*:\s*"PASS"'){throw 'Verification browser check failed.'}
