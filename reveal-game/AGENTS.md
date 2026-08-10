# 이미지 공개 퀴즈 앱 작업 지침

## 범위와 구조

- 이 디렉터리는 기존 Jekyll 테마와 분리된 정적 웹 앱이다.
- `index.html`은 진행자·단일 모니터·내부 랭킹 UI를 제공한다.
- `display.html`과 `leaderboard.html`은 각각 참가자 화면과 랭킹 전용 화면이다.
- `js/domain.js`, `js/state-machine.js`, `js/ranking.js`는 DOM에 의존하지 않는 순수 로직이다.
- `js/renderer.js`, `js/image-loader.js`, `js/persistence.js`, `js/sync.js`는 브라우저 기능을 분리한다.
- `js/app.js`, `js/display.js`, `js/leaderboard.js`만 각 페이지의 DOM을 조정한다.
- `offline.html`은 `.github/reveal-game-tests/scripts/build-offline.mjs`로 생성한다.

## 비협상 조건

- production runtime은 HTML, CSS, Vanilla JavaScript와 browser-native API만 사용한다.
- 외부 CDN, 외부 폰트, analytics, tracking, upload API를 추가하지 않는다.
- 선택한 이미지와 랭킹을 서버로 보내거나 service worker cache에 저장하지 않는다.
- 사용자 문자열을 `innerHTML`로 삽입하지 않는다.
- 모든 production 경로는 `/reveal-game/` 안의 상대 경로여야 한다.
- service worker scope와 fetch 처리는 `/reveal-game/` 밖으로 나가지 않는다.
- 배포되는 app shell 파일이 바뀌면 `sw.js`의 versioned cache 이름도 함께 올린다.
- `offline.html`을 직접 편집하지 않고 생성 스크립트와 원본을 수정한다.
- 기존 블로그 파일, `_config.yml`, 루트 workflow, CNAME/DNS 설정은 수정하지 않는다.

## 완료 검증

저장소 루트에서 다음을 실행한다.

```sh
npm ci --prefix .github/reveal-game-tests
npm test --prefix .github/reveal-game-tests
npm run build:offline --prefix .github/reveal-game-tests
npm run check:offline --prefix .github/reveal-game-tests
npm run validate --prefix .github/reveal-game-tests
npm run test:e2e --prefix .github/reveal-game-tests
PATH="/opt/homebrew/opt/ruby@3.3/bin:$PATH" bundle exec jekyll build
```

실패한 검증을 skip하거나 production 기능을 제거해 통과시키지 않는다.
