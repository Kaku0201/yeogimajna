# 지도 이미지 폴더

> **일반 사용자는 이 폴더를 직접 수정하지 않습니다.**  
> 배포 방법: [`docs/PACKAGING.md`](../../docs/PACKAGING.md)

## 폴더 구조

```
maps/
├── 신생/
│   ├── central_shroud.png      ← 지역 전체 지도 (zones.json id와 동일)
│   └── outer_la_noscea.png
├── 창천/
├── 홍련/
├── 칠흑/
├── 효월/
└── 황금/
```

- **확장팩 폴더 6개** — `zones.json` 최상위 키와 동일 (`신생`, `창천`, …)
- **파일명** — `{지역_id}.png` (예: `outer_la_noscea.png`)
- **지역당 1장** — 상세 지도 + 마커는 코드로 표시

### (선택) match 템플릿

match가 필요할 때만 같은 폴더에 `{id}_01.png` 형식으로 추가합니다.  
대부분 **detail 1장 + OCR**만으로 충분합니다.

## zones.json

지역 데이터·에테·보물 좌표는 `data/zones.json`에 작성합니다.  
→ [`data/zones.json`](../data/zones.json) · 필드 설명은 아래 참고.

| 필드 | 설명 |
|------|------|
| `aetherytes[]` | 에테 이름 + 좌표 |
| `spots[]` | 보물 좌표 + 가까운 에테 (한 쌍) |
| `detail_offset_x/y` | 상세 지도 마커 미세 보정 |

## 에테 아이콘

```
assets/aetherytes/굽은가지 목장.png
assets/treasure_marker.png
assets/Aetheryte.png   ← 기본 아이콘
```

## 캡처 방법

게임 **금 테두리 보물지도 창 전체**를 드래그합니다.  
좌표 숫자 없이 **빨간 X**만 있어도 됩니다.
