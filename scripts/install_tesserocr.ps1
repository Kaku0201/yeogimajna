# Windows용 tesserocr 미리빌드 wheel 설치 (PyPI에는 Windows wheel 없음)
# 사용: .\scripts\install_tesserocr.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$pyVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$tag = python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
$arch = python -c "import struct; print('win_amd64' if struct.calcsize('P')==8 else 'win32')"

$release = "tesserocr-v2.10.0-tesseract-5.5.2"
$wheel = "tesserocr-2.10.0-${tag}-${tag}-${arch}.whl"
$url = "https://github.com/simonflueckiger/tesserocr-windows_build/releases/download/$release/$wheel"

Write-Host "Python $pyVer ($arch) — wheel: $wheel"
Write-Host "다운로드: $url"

python -m pip install $url
if ($LASTEXITCODE -ne 0) {
    Write-Error @"
tesserocr wheel 설치 실패.
지원 wheel: https://github.com/simonflueckiger/tesserocr-windows_build/releases
Python 3.9~3.14 (win_amd64/win32). 실패 시 pytesseract fallback으로 앱은 동작합니다.
"@
    exit 1
}

python -c "import tesserocr; print('tesserocr OK', tesserocr.__version__)"
if ($LASTEXITCODE -ne 0) { exit 1 }
