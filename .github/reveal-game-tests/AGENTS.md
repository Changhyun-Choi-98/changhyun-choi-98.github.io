# 이미지 공개 퀴즈 테스트 지침

## 목적

- Node 내장 test runner로 상태 머신, 타이머, 공개 수학, 랭킹, 파일 정렬을 검증한다.
- Playwright로 실제 DOM, Canvas, popup, storage, service worker, offline 동작을 검증한다.
- fixture는 이 디렉터리에서 생성한 합성 이미지만 사용한다.
- screenshot과 임시 test result는 commit하지 않는다.

## 명령

```sh
npm test
npm run build:offline
npm run check:offline
npm run validate
npm run test:e2e
npm run test:webkit
```

## 제약

- 실제 사용자 이미지나 경로를 fixture에 포함하지 않는다.
- 외부 network가 필요한 production 코드를 허용하지 않는다.
- service worker가 `/reveal-game/` 밖을 제어하거나 cache하도록 만들지 않는다.
- `offline.html` 재생성 결과가 Git diff를 만들면 source 또는 artifact를 함께 바로잡는다.
- 실패 test를 skip하지 않는다.

## 완료 기준

- unit, static validation, offline reproducibility, Chromium/WebKit E2E가 통과한다.
- Jekyll `_site`에 앱 필수 파일이 생성되고 `.github/reveal-game-tests`는 노출되지 않는다.
- 콘솔 오류, page error, 예상 밖 외부 요청이 없다.
