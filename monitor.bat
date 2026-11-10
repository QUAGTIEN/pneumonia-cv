@echo off
rem Giong SERIAL MONITOR cua Arduino IDE. Thoat bang Ctrl + C.
cd /d "%~dp0firmware"
"%USERPROFILE%\.platformio\penv\Scripts\pio.exe" device monitor
