# 이미지 공개 퀴즈 배포 안내

## 기능 브랜치와 PR

1. `BASE_COMMIT`에서 만든 `feature/image-reveal-game` 브랜치에서만 작업한다.
2. unit, 정적 검증, standalone 재현성, Chromium/WebKit, Jekyll build를 모두 통과시킨다.
3. 변경 경로가 `reveal-game/**`, `.github/reveal-game-tests/**`, `.github/workflows/reveal-game-ci.yml`뿐인지 확인한다.
4. 논리적인 commit을 만들고 기능 브랜치를 origin에 push한다.
5. base가 `main`인 PR을 만들고 기능, 구조, 개인정보, 실제 시험 결과, rollback을 본문에 기록한다.
6. Windows, macOS, Linux 전용 CI가 모두 성공할 때까지 원인을 수정하고 다시 실행한다.

기능 브랜치 push와 PR은 기존 Pages workflow의 `main` 조건을 만족하지 않으므로 production 배포를 일으키지 않는다.

## Production 승인 게이트

PR과 CI가 준비되어도 `main`으로 merge하지 않는다. 사용자에게 다음 문구로 승인을 요청한다.

> Production GitHub Pages에 배포하려면 DEPLOY라고 입력해 주세요.

정확한 `DEPLOY` 승인 전에는 merge, `main` push, Pages 수동 실행을 하지 않는다.

## 승인 후 배포

1. origin의 최신 `main`을 fetch하고 기능 브랜치와 충돌 여부를 확인한다.
2. PR head와 모든 required CI가 최신 commit에서 green인지 다시 확인한다.
3. 허용 경로 밖 diff와 사용자 파일이 없는지 확인한다.
4. 반드시 squash merge하여 production 변경을 하나의 commit으로 만들고 생성된 hash를 `DEPLOY_COMMIT`으로 기록한다. 다른 merge 방식은 이 배포 절차에서 사용하지 않는다.
5. 기존 `.github/workflows/pages.yml`이 시작되고 성공할 때까지 확인한다.
6. 아래 production 주소를 실제 브라우저에서 검사한다.

- `https://changhyunchoi.com/`
- 기존 주요 blog page
- `https://changhyunchoi.com/reveal-game/`
- `/reveal-game/display.html`
- `/reveal-game/leaderboard.html`
- `/reveal-game/help.html`
- manifest, icon, service worker와 `/reveal-game/` 한정 scope

7. 오프라인 준비 완료, 폴더 선택, 모자이크, 검은색 원, 참가자 팝업, 랭킹 저장을 smoke test한다.
8. 최종 보고서에 실제 `DEPLOY_COMMIT`과 Pages 실행 URL을 기록한다.

## 실패와 원상 복구

Pages가 실패하면 완료라고 보고하지 않는다. 원인을 feature/후속 수정 commit에서 고치고 정상 배포를 확인한다. 앱을 제거할 때는 `ROLLBACK.md`의 `git revert <DEPLOY_COMMIT>` 절차를 사용하며 강제 push나 hard reset을 사용하지 않는다.
