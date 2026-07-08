# 여기맞나?

FFXIV 게임 위에 표시되는 보물지도 좌표 인식 오버레이

## 실행 방법 (개발)

```powershell
cd d:\cording\TR
python -m pip install -r requirements.txt
python main.py
```

개발용 상세 로그: `TR_DEBUG=1 python main.py`

## 배포 빌드 (Windows)

```powershell
.\scripts\build_release.ps1
```

빌드 후 `dist\yeogimajna-win64-v*.zip` 이 생성됩니다.

**사용자 흐름:** zip 다운로드 → 압축 해제 → `여기맞나.exe` 실행 (설치 불필요)

> `build\yeogimajna\` 안의 exe는 중간 파일이라 실행하면 `python314.dll` 오류가 납니다.

자세한 내용은 [`docs/PACKAGING.md`](docs/PACKAGING.md) 참고.

## 기능

| 기능 | 설명 |
|------|------|
| 플로팅 버튼 | 항상 위에 표시, 드래그 이동, 위치 자동 저장 |
| 좌클릭 | 인식 박스 표시 → 확인 버튼으로 지도 분석 |
| 인식 결과 | 지역명, 좌표, 가장 가까운 에테라이트 |
| 후보 확정 | ref 후보 3개 중 선택·학습 캐시로 재인식 가속 |
| 좌표 클릭 | 상세 지도 (빨간 원으로 위치 표시) |
| 결과 우클릭 | 결과 삭제 (버튼만 남음) |
| 버튼 우클릭 | 고정/해제, 투명도, 종료 |

## 지도 데이터 (사용자)

**일반 사용자는 지도 PNG를 직접 추가하지 않습니다.**

설치 파일에 **지도 팩**이 포함되어 있거나, 첫 실행 시 자동으로 다운로드됩니다.  
준비가 끝나면 🗺️ 버튼 → 보물지도 창 드래그만 하면 됩니다.

## OCR (Tesseract)

배포 zip에 **Tesseract OCR + 한글 데이터가 포함**됩니다. 사용자는 별도 설치가 필요 없습니다.

개발 중 시스템 Tesseract를 쓰려면 `winget install UB-Mannheim.TesseractOCR` 로 설치할 수 있습니다.

## 설정·학습 저장

| 파일 | 내용 |
|------|------|
| `%USERPROFILE%\.yeogimajna\settings.json` | 버튼 위치, 투명도, 고정 상태 |
| `%USERPROFILE%\.yeogimajna\learned_refs.json` | **확정한 보물지도 학습** (앱 꺼도 유지) |

확정(✓)한 지도는 지형 지문이 저장됩니다. 다음 실행 때 같은 지도를 캡처하면 ref 전체 스캔 없이 **⚡ 학습 매칭**으로 바로 좌표가 나옵니다.

> zip을 새로 풀어도 학습 데이터는 **PC 사용자 폴더**에 남습니다.  
> 횟수가 이상하면 인식 결과 패널 하단 **「학습 데이터 초기화」** 버튼을 누르세요.

구버전 `%USERPROFILE%\.tr_overlay` 가 있으면 자동 호환됩니다.
