set PROJECT_NAME=main
set SIGNTOOL_PATH="setup/signtool.exe"
set CERT_PATH="setup/mycert.pfx"
set TIMESTAMP_SERVER=http://timestamp.digicert.com
set EXEC_NAME="dist/LDV001.exe"
set SETUP_NAME="setup/LDV001Setup.exe"
SET NSI_NAME="setup/main.nsi"
SET NSIS_TOOL="C:\Program Files (x86)\NSIS\makensis.exe"

REM INSTALL
::pyinstaller %PROJECT_NAME%.spec

REM PYARMOR
pyarmor gen --pack %PROJECT_NAME%.spec -r pano_blur.py pano_blur_gui.py

::REM PFX CODESIGN
%SIGNTOOL_PATH% sign /a /s MY /n "KnWorks Co., Ltd." /fd sha256 /tr http://rfc3161timestamp.globalsign.com/advanced /td SHA256 %EXEC_NAME%

REM BUILD
%NSIS_TOOL% %NSI_NAME%

REM SHA256
%SIGNTOOL_PATH% sign /a /s MY /n "KnWorks Co., Ltd." /fd sha256 /tr http://rfc3161timestamp.globalsign.com/advanced /td SHA256 %SETUP_NAME%

