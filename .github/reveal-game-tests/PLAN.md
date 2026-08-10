# 이미지 공개 퀴즈 구현 계획

## 기준과 변경 경계

- 저장소: `Changhyun-Choi-98/changhyun-choi-98.github.io`
- 작업 브랜치: `feature/image-reveal-game`
- BASE_COMMIT: `99309db2a41e2d8fcd60e16dc479af50c0a63657`
- 허용 경로: `reveal-game/**`, `.github/reveal-game-tests/**`, `.github/workflows/reveal-game-ci.yml`
- production merge는 사용자의 정확한 `DEPLOY` 승인 전까지 수행하지 않는다.

## 1. 저장소 조사

- Jekyll 4.4.1, Just the Docs 0.12.0, Ruby 3.3 CI를 유지한다.
- 기존 Pages workflow는 `main` push에서만 배포하므로 수정하지 않는다.
- 앱은 front matter 없는 정적 파일로 추가해 기존 theme/header/sidebar를 상속하지 않는다.
- dot-directory인 `.github/reveal-game-tests`가 `_site`에 나오지 않는지 build 후 검사한다.

## 2. Architecture

- 순수 domain: 시간 포맷/파싱, easing, mosaic/circle 수학, 자연 정렬.
- reducer: 명시적 `EMPTY`, `READY`, `RUNNING`, `PAUSED`, `REVEALED_PENDING_RESULT`, `FINALIZED`, `SESSION_COMPLETE` 전이.
- renderer: DPR과 pixel budget을 적용한 Canvas 2D contain renderer.
- image loader: picker/input/drop 재귀 수집, current+next 한정 decode, Object URL 정리.
- persistence: versioned IndexedDB, localStorage 정상 snapshot, memory fallback.
- synchronization: BroadcastChannel, opener `postMessage`, storage scalar snapshot 순서.
- UI adapter: 진행자, 참가자, 랭킹 전용 페이지별 얇은 DOM 계층.

## 3. 상태 머신과 타이머

- reducer만 결과 집계와 상태 전이를 변경한다.
- stopwatch는 주입된 monotonic `now`로 active elapsed를 계산한다.
- confirmation modal은 running round를 일시 정지하고, 취소 시 같은 elapsed에서 재개한다.
- finalized round만 session 합계에 고정하고 현재 active round를 별도로 더한다.
- duration 도달 시 elapsed를 정확히 clamp하고 판정 대기 상태로 이동한다.

## 4. 데이터 스키마

- IndexedDB v1 stores: `rankings`, `settings`, `snapshots`.
- ranking: `id`, `teamName`, `normalizedTeamName`, `correctCount`, `elapsedMs`, `createdAt`, `updatedAt`.
- setting: 공개 mode, duration, speed profile, image order.
- image Blob과 file metadata는 저장하지 않는다.
- JSON backup은 `schemaVersion`, `exportedAt`, `rankings`, `settings`를 검증한다.

## 5. Renderer 수학

- easing: slow `p^1.65`, balanced `p^1.20`, fast `p^0.75`.
- mosaic: 1×1 endpoint, log-spaced unique levels, 원본 endpoint, 단계 역행 방지.
- downsample은 smoothing high와 단계 축소, upscale은 nearest-neighbor를 사용한다.
- circle: contain rectangle 중심, `r0 = diagonal/2 + margin`, `r = r0*sqrt(1-e)`, image rect clip.
- resize는 ResizeObserver에서 animation frame 단위로 병합한다.

## 6. PWA와 standalone offline

- manifest start URL/scope는 `./`, service worker는 자기 scope의 allowlist만 cache한다.
- cache 완료 응답을 받은 뒤에만 오프라인 준비 완료를 표시한다.
- update waiting 상태에서 사용자 버튼으로만 skipWaiting/reload한다.
- `offline.html`은 동일 CSS/JS를 inline하는 deterministic builder 결과로 관리한다.
- file URL에서는 service worker 없이 동작하고 participant popup은 about:blank 직접 구성을 시도한다.

## 7. Testing

- unit: reducer, fake monotonic clock, mosaic/circle, ranking/CSV/JSON, image filtering/order.
- E2E: folder fixture, gameplay, modal cancel/confirm, duration, ranking CRUD/import/export/persistence, popup sync.
- offline: service worker install/cache/offline reload, standalone file URL.
- viewport: 1366×768, 1920×1080, 3840×2160의 controller/display/leaderboard screenshot.
- integration: Jekyll build 산출물, 기존 root/index 주요 페이지, 링크와 scope를 검사한다.

## 8. 배포와 rollback

- 전체 local 검증 후 논리적 commit을 feature branch에 push하고 `main` 대상 PR을 만든다.
- 별도 CI matrix에서 Windows, macOS, Linux 결과를 확인하고 실패를 수정한다.
- green 상태에서 `DEPLOY` 승인을 요청하고 그 전에는 merge하지 않는다.
- 배포 후 rollback은 기록한 squash `DEPLOY_COMMIT`을 `git revert`하는 방식만 안내한다.
- PWA cache와 로컬 ranking backup/삭제 절차를 한국어 guide에 포함한다.
