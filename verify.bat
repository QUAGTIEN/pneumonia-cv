@echo off
rem Giong nut VERIFY cua Arduino IDE: bien dich firmware, khong nap.
cd /d "%~dp0firmware"
"%USERPROFILE%\.platformio\penv\Scripts\pio.exe" run
