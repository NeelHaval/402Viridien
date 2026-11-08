$ErrorActionPreference='Stop'
$lane=$PSScriptRoot
$root='C:/Users/hssil/Downloads/03_Programming_Projects/hacktheburgh/viridien_work'
$dataset="$root/external_unlabelled_test"
$python='C:/Users/hssil/anaconda3/python.exe'
Set-Location -LiteralPath "$root/bakeoff"
& $python -X utf8 "$lane/run_suite.py" $dataset "$lane/runs/clean" --workers 2 *> "$lane/clean_run.log"
if ($LASTEXITCODE -ne 0) { throw "Clean run failed: exit $LASTEXITCODE" }
& $python -I -X utf8 "$lane/make_degraded.py" $root $dataset *> "$lane/degrade_build.log"
if ($LASTEXITCODE -ne 0) { throw "Scan copy failed: exit $LASTEXITCODE" }
& $python -X utf8 "$lane/run_suite.py" "$dataset/degraded" "$lane/runs/medium" --workers 2 --clean-dir "$lane/runs/clean" *> "$lane/medium_run.log"
if ($LASTEXITCODE -ne 0) { throw "Scan run failed: exit $LASTEXITCODE" }
Write-Output 'Completed clean and fixed medium scan runs.'
