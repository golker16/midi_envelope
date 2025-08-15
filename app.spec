# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

block_cipher = None

project_dir = Path('.')
assets_dir = project_dir / 'assets'

datas = []
if assets_dir.exists():
    datas += [
        (str(assets_dir / 'app.png'), 'assets'),
        (str(assets_dir / 'app.ico'), 'assets'),
    ]

hiddenimports = []

a = Analysis(
    ['app.py'],
    pathex=[str(project_dir.resolve())],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CopyEnvelope',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # ventana GUI (sin consola)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(assets_dir / 'app.ico') if assets_dir.exists() else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CopyEnvelope'
)
