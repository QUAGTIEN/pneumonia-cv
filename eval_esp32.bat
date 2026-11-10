@echo off
rem Buoc 6: gui 624 anh test xuong ESP32 qua USB, ghi results\esp32_report.txt
rem Tat Serial Monitor truoc.   .\eval_esp32.bat --limit 20  = chay thu 20 anh
cd /d "%~dp0"
"venv-tf\Scripts\python.exe" scripts\6_esp32_eval.py %*
