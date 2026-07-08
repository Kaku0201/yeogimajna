# 배포 가이드 (개발자용)

앱 이름: **여기맞나?** (`yeogimajna`)

일반 사용자는 **지도 PNG를 직접 추가하지 않습니다.**  
개발팀이 한 번 지도 팩을 만들어 설치 파일에 넣거나, 첫 실행 시 자동 다운로드합니다.

## 사용자 경험

1. `dist\여기맞나` 폴더 배포 (또는 설치 프로그램)
2. (지도 팩 포함 시) 바로 🗺️ 버튼 사용
3. (경량 설치 시) 첫 실행에 지도 데이터 자동 다운로드
4. 게임 보물지도 창 드래그 → 좌표 표시

## 빌드

```powershell
.\scripts\build_release.ps1
```

- 출력 폴더: `dist\여기맞나\여기맞나.exe`
- **배포 zip**: `dist\yeogimajna-win64-v1.0.0.zip` (버전은 manifest 기준)
- `data/`, `assets/` 전체가 번들에 포함됩니다.
- 배포 전 `assets/maps/`에 지역 PNG, `assets/treasure_refs/`에 ref 이미지를 채워 넣으세요.

### zip 포터블 배포 (권장)

설치 프로그램 없이 **압축 해제 후 바로 실행**하는 방식입니다. 이미 이 형태로 빌드됩니다.

1. `yeogimajna-win64-v*.zip` 을 사용자에게 전달
2. 사용자가 원하는 폴더에 압축 해제
3. `여기맞나.exe` 더블클릭

폴더 안에 exe와 `_internal` 이 같이 있어야 합니다. exe만 따로 복사하면 안 됩니다.

### 주의: build 폴더 exe는 실행하지 마세요

PyInstaller는 중간 파일을 `build\yeogimajna\`에 두지만, **실행 가능한 배포본은 `dist\여기맞나\`만**입니다.

| 경로 | 용도 |
|------|------|
| `build\yeogimajna\여기맞나.exe` | 중간 산출물 — `_internal` 없음 → **실행 불가** |
| `dist\여기맞나\여기맞나.exe` | 배포용 — `_internal\python314.dll` 포함 |

`Failed to load Python DLL` / `python314.dll` 오류가 나면 `dist` 폴더의 exe를 실행했는지 확인하세요.

## 지도 팩 구조

```
assets/maps/
├── 신생/
│   ├── central_shroud.png
│   └── outer_la_noscea.png
├── 창천/
├── 홍련/
├── 칠흑/
├── 효월/
└── 황금/
```

- **확장팩 폴더** — `zones.json` 키와 동일 (6개)
- **파일명** — `{zone_id}.png` (지역당 1장)
- **zones.json** — 에테·보물 좌표 (함께 배포)

## 설치 파일에 포함 (권장)

```
여기맞나/
├── 여기맞나.exe
├── data/
│   ├── zones.json
│   ├── treasure_ref_coords/
│   └── map_pack_manifest.json
└── assets/
    ├── maps/
    ├── treasure_refs/
    └── aetherytes/
```

## 경량 설치 + 자동 다운로드

`data/map_pack_manifest.json`:

```json
{
  "version": "1.0.0",
  "download_url": "https://your-cdn/yeogimajna-map-pack-1.0.0.zip",
  "min_detail_maps": 50,
  "min_match_templates": 0
}
```

다운로드 위치: `%USERPROFILE%\.yeogimajna\map_pack`

## Tesseract (OCR)

배포 zip에 **Tesseract 실행 파일 + tessdata(eng/osd/kor)** 가 포함됩니다.

빌드 시 `scripts/stage_tesseract.ps1` 가 개발 PC의 Tesseract 설치본을 `vendor/tesseract` 로 복사하고, `kor.traineddata` 가 없으면 GitHub에서 받습니다.

**개발자 PC (빌드용):** `winget install UB-Mannheim.TesseractOCR` — 빌드 머신에만 필요, 최종 사용자는 불필요.

**사용자:** zip 압축 해제 후 바로 OCR 사용 가능.

## 로깅·디버그

| 환경 변수 | 용도 |
|-----------|------|
| (없음) | 배포 기본 — WARNING 이상만, 디버그 이미지 미저장 |
| `TR_DEBUG=1` | 상세 로그 + `TR_DEBUG_QUERY` 동작 |
| `TR_DEBUG_QUERY=1` | 매칭 시 `debug_query.png` 저장 |
| `TR_QC_DEBUG_DIR=경로` | 캡처 품질 검사 프레임 저장 |

## 이미지 매칭

- ref DB: terrain SSIM + coarse 가속
- detail fallback: ORB → NCC (ref 없을 때)
