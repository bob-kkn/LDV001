# -*- mode: python ; coding: utf-8 -*-
import sys, pathlib

pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
candidates = [
    pathlib.Path(sys.prefix) / f"Frameworks/Python.framework/Versions/{pyver}/Python",
    pathlib.Path(f"/Library/Frameworks/Python.framework/Versions/{pyver}/Python"),
    pathlib.Path(f"/opt/homebrew/Frameworks/Python.framework/Versions/{pyver}/Python"),
    pathlib.Path(f"/usr/local/Frameworks/Python.framework/Versions/{pyver}/Python"),
]
python_lib = next((p for p in candidates if p.exists()), None)
if not python_lib:
    raise SystemExit(f"Could not locate Python.framework/Versions/{pyver}/Python")
if python_lib.is_symlink():
    python_lib = python_lib.resolve()


block_cipher = None

a = Analysis(
    ['pano_blur_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('./Resource/*', './Resource'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
extra_bins = [('Python', str(python_lib), 'BINARY')]

exe = EXE(
    pyz,
    a.scripts,
    a.binaries + extra_bins,
    a.zipfiles,
    a.datas,
    name='LDV001',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    target_arch="arm64",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='cau104_collected'
)

app = BUNDLE(
    exe,
    name='LDV001.app',
    icon='Resource/icon.icns',
    bundle_identifier='kr.KnWorks.LDV001',
    info_plist={
        'CFBundleName': 'LDV001',
        'CFBundleDisplayName': 'LDV001',
        'CFBundleShortVersionString': '1.0.0.0',
        'CFBundleVersion': '1.0.0.0',
        'NSHumanReadableCopyright': 'Copyright © KnWorks. All rights reserved.'
    },
)
