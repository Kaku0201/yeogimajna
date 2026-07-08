# 여기맞나? Windows 배포 빌드
# 사용: .\scripts\build_release.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "의존성 설치..."
python -m pip install -r requirements.txt -r requirements-build.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "tesserocr (고속 OCR) 설치..."
powershell -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\install_tesserocr.ps1")
if ($LASTEXITCODE -ne 0) {
    Write-Warning "tesserocr 설치 실패 — 배포본은 pytesseract fallback으로 동작합니다 (OCR 느림)"
}

Write-Host "Tesseract 번들 준비..."
powershell -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\stage_tesseract.ps1")

Write-Host "앱 아이콘 준비..."
python (Join-Path $Root "scripts\prepare_icon.py")

Write-Host "PyInstaller 빌드..."
python -m PyInstaller yeogimajna.spec --noconfirm --clean

$DistRoot = Join-Path $Root "dist"
$OutDir = Get-ChildItem $DistRoot -Directory -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $OutDir) {
    Write-Error "빌드 출력 폴더를 찾을 수 없습니다: $DistRoot"
}

$Exe = Get-ChildItem $OutDir.FullName -Filter "*.exe" -File |
    Where-Object { $_.DirectoryName -eq $OutDir.FullName } |
    Select-Object -First 1

if (-not $Exe) {
    Write-Error "실행 파일을 찾을 수 없습니다: $($OutDir.FullName)"
}

$Dll = Join-Path $OutDir.FullName "_internal\python314.dll"
if (-not (Test-Path $Dll)) {
    Write-Error "python DLL이 없습니다. 빌드가 불완전합니다: $Dll"
}

$Internal = Join-Path $OutDir.FullName "_internal"
$TesserocrBundled = @(Get-ChildItem $Internal -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "tesserocr" }).Count -gt 0
if ($TesserocrBundled) {
    Write-Host "tesserocr: 배포본에 포함됨 (고속 OCR)" -ForegroundColor Green
} else {
    Write-Warning "tesserocr: 배포본에 미포함 — pytesseract fallback 사용"
}

$Readme = @"
여기맞나? — 실행 방법

1. 압축을 풀면 폴더가 나옵니다.
2. 폴더 안의 $($Exe.Name) 를 더블클릭하세요.
3. _internal 폴더는 지우거나 옮기지 마세요 (exe와 같은 위치에 있어야 합니다).

설치 프로그램은 필요 없습니다. 압축 해제 후 바로 실행하면 됩니다.
OCR(Tesseract tessdata + tesserocr 고속 엔진)이 zip에 포함되어 별도 설치가 필요 없습니다.
"@

$Readme | Out-File -FilePath (Join-Path $OutDir.FullName "README.txt") -Encoding utf8

$Version = "0.0.0"
$ManifestPath = Join-Path $Root "data\map_pack_manifest.json"
if (Test-Path $ManifestPath) {
    try {
        $manifest = Get-Content $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($manifest.version) { $Version = [string]$manifest.version }
    } catch { }
}

$ZipName = "yeogimajna-win64-v$Version.zip"
$ZipPath = Join-Path $DistRoot $ZipName
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }

Write-Host "배포 zip 생성: $ZipName ..."
Compress-Archive -Path $OutDir.FullName -DestinationPath $ZipPath -CompressionLevel Optimal

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " 빌드 완료" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "실행 파일:" $Exe.FullName
Write-Host "배포 zip  :" $ZipPath
Write-Host ""
Write-Host "사용자: zip 압축 해제 → $($Exe.Name) 실행 (설치 불필요)" -ForegroundColor Cyan
Write-Host ""
Write-Host "[개발자] build\yeogimajna\ exe는 중간 파일입니다. dist 또는 zip만 배포하세요." -ForegroundColor Yellow
Write-Host ""

Start-Process explorer.exe $DistRoot
