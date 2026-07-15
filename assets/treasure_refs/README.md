# spot 참조 PNG (인게임 UI 스타일)

```
assets/treasure_refs/{확장팩}/{zone_id}/24.05_10.79.png
```

- **확장팩 폴더** — `maps/`와 동일 (`신생`, `창천`, `홍련`, `칠흑`, `효월`, `황금`)
- **지역 폴더** — `zones.json`의 `id` (예: `central_shroud`)
- **파일명** — spots treasure 좌표 `X_Y.png`

그리드 분할:

```bash
python scripts/split_treasure_grid.py <그리드.png> <zone_id>
```

## 준비됨

- `신생/central_shroud/` — 21장 (검은장막 숲 중부 삼림)
- `신생/east_shroud/` — 21장 (검은장막 숲 동부 삼림)
- `신생/south_shroud/` — 21장 (검은장막 숲 남부 삼림)
- `신생/north_shroud/` — 21장 (검은장막 숲 북부 삼림)
- `신생/central_la_noscea/` — 21장 (중부 라노시아)
- `신생/lower_la_noscea/` — 21장 (저지 라노시아)
- `신생/eastern_la_noscea/` — 21장 (동부 라노시아)
- `신생/western_la_noscea/` — 21장 (서부 라노시아)
- `신생/upper_la_noscea/` — 21장 (고지 라노시아)
- `신생/outer_la_noscea/` — 21장 (외지 라노시아)
- `신생/western_thanalan/` — 21장 (서부 다날란)
- `신생/central_thanalan/` — 21장 (중부 다날란)
- `신생/eastern_thanalan/` — 21장 (동부 다날란)
- `신생/southern_thanalan/` — 21장 (남부 다날란)
- `신생/northern_thanalan/` — 10장 (북부 다날란)
- `신생/coerthas_central_highlands/` — 19장 (커르다스 중앙고지)
- `신생/mor_dhona/` — 13장 (모르도나)

**신생 에오르제아 17지역 준비 완료**

## 창천

- `창천/coerthas_western_highlands/` — 22장 (커르다스 서부고지)
- `창천/the_dravanian_hinterlands/` — 26장 (저지 드라바니아)
- `창천/the_dravanian_forelands/` — 8장 (고지 드라바니아)
- `창천/the_churning_mists/` — 26장 (드라바니아 구름바다)
- `창천/the_sea_of_clouds/` — 20장 (아발라시아 구름바다)

## 홍련

- `홍련/yanxia/` — 20장 (얀샤) · 5×5 그리드 중 5번째 줄(8인 중복) 제외
- `홍련/the_fringes/` — 16장 (기라바니아 변경지대) · 24칸 중 4~5줄(8인 중복) 제외
- `홍련/the_peaks/` — 16장 (기라바니아 산악지대) · 24칸 중 4~5줄(8인 중복) 제외
- `홍련/the_lochs/` — 20장 (기라바니아 호반지대) · 5×5 그리드 중 5번째 줄(8인 중복) 제외
- `홍련/the_ruby_sea/` — 21장 (홍옥해) · 5×5 그리드 중 5번째 줄 2~5칸(8인 중복) 제외
- `홍련/the_azim_steppe/` — 18장 (아짐 대초원) · 26칸 중 4~5줄(8인 중복) 제외

**홍련 7지역 준비 완료**

## 칠흑

- `칠흑/lakeland/` — 16장 (레이크랜드) · 24칸 중 4~5줄(8인 중복) 제외
- `칠흑/kholusia/` — 16장 (콜루시아 섬) · 24칸 중 4~5줄(8인 중복) 제외
- `칠흑/amh_araeng/` — 16장 (아므 아랭) · 24칸 중 4~5줄(8인 중복) 제외
- `칠흑/il_mheg/` — 16장 (일 메그) · 24칸 중 4~5줄(8인 중복) 제외
- `칠흑/the_raktika_greatwood/` — 16장 (라케티카 대삼림) · 24칸 중 4~5줄(8인 중복) 제외
- `칠흑/the_tempest/` — 16장 (템페스트) · 24칸 중 4~5줄(8인 중복) 제외

**칠흑 6지역 준비 완료**

## 효월

- `효월/labyrinthos/` — 16장 (라비린토스) · 24칸 중 4~5줄(8인 중복) 제외
- `효월/thavnair/` — 16장 (사베네어 섬) · 24칸 중 4~5줄(8인 중복) 제외
- `효월/garlemald/` — 16장 (갈레말드) · 24칸 중 4~5줄(8인 중복) 제외
- `효월/mare_lamentorum/` — 16장 (비탄의 바다) · 24칸 중 4~5줄(8인 중복) 제외
- `효월/ultima_thule/` — 16장 (울티마 툴레) · 24칸 중 4~5줄(8인 중복) 제외
- `효월/elpis/` — 8장 (엘피스) · 16칸 중 2~3줄(8인 중복) 제외

**효월 7지역 준비 완료**

## 황금

- `황금/urqopacha/` — 16장 (오르코 파차)
- `황금/kozamauka/` — 16장 (코자말루 카)
- `황금/yak_tel/` — 16장 (야크텔 밀림)
- `황금/shaaloani/` — 16장 (샬로니 황야)
- `황금/heritage_found/` — 16장 (헤리티지 파운드)
- `황금/living_memory/` — 8장 (리빙 메모리) · 6+2칸 그리드 (8인만)

**황금 6지역 준비 완료**

## 그리드 보낼 때

- `zones.json` spots **개수·순서**(왼→오, 위→↓)가 그리드와 같아야 합니다.
