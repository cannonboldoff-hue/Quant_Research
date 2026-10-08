# Stops running ablation processes (chain B driver, ablation script and its workers).
$me = $PID
Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $me -and $_.CommandLine -match 'tfml_finish_b|tfml_ablation' -and $_.CommandLine -notmatch 'Get-CimInstance'
} | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
