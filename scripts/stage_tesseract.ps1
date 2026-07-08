# Tesseract OCR 런타임을 vendor/tesseract 에 스테이징 (배포 번들용)
# 사용: .\scripts\stage_tesseract.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Dest = Join-Path $Root "vendor\tesseract"
$TessData = Join-Path $Dest "tessdata"

$SourceCandidates = @(
    "${env:ProgramFiles}\Tesseract-OCR",
    "${env:ProgramFiles(x86)}\Tesseract-OCR"
)

$Source = $SourceCandidates | Where-Object { Test-Path (Join-Path $_ "tesseract.exe") } | Select-Object -First 1
if (-not $Source) {
    Write-Error @"
Tesseract OCR이 이 PC에 없습니다. 빌드 전 한 번 설치하세요:

  winget install UB-Mannheim.TesseractOCR

설치 후 이 스크립트를 다시 실행하세요.
"@
}

Write-Host "Tesseract 스테이징: $Source -> $Dest"

if (Test-Path $Dest) {
    Remove-Item $Dest -Recurse -Force
}
New-Item -ItemType Directory -Path $TessData -Force | Out-Null

Copy-Item (Join-Path $Source "tesseract.exe") $Dest
Get-ChildItem $Source -Filter "*.dll" | Copy-Item -Destination $Dest

$LangFiles = @("eng.traineddata", "osd.traineddata", "kor.traineddata")
foreach ($name in $LangFiles) {
    $srcFile = Join-Path $Source "tessdata\$name"
    $dstFile = Join-Path $TessData $name
    if (Test-Path $srcFile) {
        Copy-Item $srcFile $dstFile
        continue
    }
    if ($name -eq "kor.traineddata" -and -not (Test-Path $dstFile)) {
        $url = "https://github.com/tesseract-ocr/tessdata/raw/main/kor.traineddata"
        Write-Host "한글 tessdata 다운로드: $url"
        Invoke-WebRequest -Uri $url -OutFile $dstFile -UseBasicParsing
    }
}

foreach ($required in @("eng.traineddata", "osd.traineddata", "kor.traineddata")) {
    if (-not (Test-Path (Join-Path $TessData $required))) {
        Write-Error "필수 tessdata 없음: $required"
    }
}

$sizeMb = [math]::Round((Get-ChildItem $Dest -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
Write-Host "완료: vendor\tesseract ($sizeMb MB)"
