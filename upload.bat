@echo off
rem Giong nut UPLOAD cua Arduino IDE: bien dich + nap vao board, roi mo Serial Monitor (thoat: Ctrl + C).
cd /d "%~dp0firmware"
"%USERPROFILE%\.platformio\penv\Scripts\pio.exe" run -t upload -t monitor
