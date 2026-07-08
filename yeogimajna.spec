# -*- mode: python ; coding: utf-8 -*-
"""여기맞나? Windows 배포 빌드 (PyInstaller)"""

from pathlib import Path

ROOT = Path(SPECPATH)

block_cipher = None

datas = [
    (str(ROOT / "data"), "data"),
    (str(ROOT / "assets"), "assets"),
]
_tess = ROOT / "vendor" / "tesseract" / "tesseract.exe"
if _tess.is_file():
    datas.append((str(ROOT / "vendor" / "tesseract"), "vendor/tesseract"))

_icon = ROOT / "assets" / "waymarks.ico"
if not _icon.is_file():
    _icon = ROOT / "assets" / "waymarks.png"
_exe_icon = str(_icon) if _icon.is_file() else None

_binaries: list = []
_hiddenimports = [
    "cv2",
    "numpy",
    "PIL",
    "pytesseract",
    "tesserocr",
    "mss",
]
try:
    from PyInstaller.utils.hooks import collect_all

    _tess_datas, _tess_binaries, _tess_hidden = collect_all("tesserocr")
    datas += _tess_datas
    _binaries = _tess_binaries
    _hiddenimports += _tess_hidden
except Exception:
    pass

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=_binaries,
    datas=datas,
    hiddenimports=_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="여기맞나",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=_exe_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="여기맞나",
)
