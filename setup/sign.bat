SET SIGN_TOOL="C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\signtool.exe"
SET INSTALLER_PATH=%~1

REM SHA1
%SIGN_TOOL% sign /a /s MY /n "KnWorks Co., Ltd." /tr http://rfc3161timestamp.globalsign.com/advanced /td SHA256 /fd SHA1 %INSTALLER_PATH%

REM SHA256
%SIGN_TOOL% sign /a /s MY /n "KnWorks Co., Ltd." /as /fd sha256 /tr http://rfc3161timestamp.globalsign.com/advanced /td SHA256 %INSTALLER_PATH%
