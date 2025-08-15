# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

block_cipher = None

project_dir = Path('.')
assets_dir = project_dir / 'assets'
molds_dir = assets_dir / 'molds'

datas = []
if assets_dir.exists():
    # iconos
    if (assets_dir / 'app.png').exists():
        datas.append((str(assets_dir / 'app.png'), 'assets'))
    if (assets_dir / 'app.ico').exists():
        datas.append((str(assets_dir / 'app.ico'), 'assets'))
# incluir todos los JSON de assets/molds
if molds_dir.exists():
    for p in molds_dir.glob('*.json'):
        datas.append((str(p), 'assets/molds'))

a = Analysis(
    ['app.py'],
    pathex=[str(project_dir.resolve())],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='CopyEnvelope',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(assets_dir / 'app.ico') if (assets_dir / 'app.ico').exists() else None,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=True, upx_exclude=[], name='CopyEnvelope'
)

