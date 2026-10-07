# Prints the number of live python processes running scripts/tfml_run.py
@(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*tfml_run.py*' }).Count
