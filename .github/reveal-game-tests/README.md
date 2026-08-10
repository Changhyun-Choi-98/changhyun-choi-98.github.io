# 이미지 공개 퀴즈 개발·검증

이 디렉터리는 production runtime에 포함되지 않는 Node/Playwright 검증 도구다. 실제 앱은 `reveal-game/`의 정적 HTML, CSS, JavaScript만으로 실행된다.

## macOS 로컬 실행

저장소 루트에서 다음을 실행한다.

```sh
npm ci --prefix .github/reveal-game-tests
npm run generate:icons --prefix .github/reveal-game-tests
npm run build:offline --prefix .github/reveal-game-tests
npm test --prefix .github/reveal-game-tests
npm run check:offline --prefix .github/reveal-game-tests
npm run validate --prefix .github/reveal-game-tests
npx --prefix .github/reveal-game-tests playwright install chromium webkit
npm run test:e2e --prefix .github/reveal-game-tests
npm run test:webkit --prefix .github/reveal-game-tests
PATH="/opt/homebrew/opt/ruby@3.3/bin:$PATH" bundle exec jekyll build
npm run validate:jekyll --prefix .github/reveal-game-tests
```

로컬 화면을 직접 열려면 다음을 실행하고 `http://127.0.0.1:4173/reveal-game/`에 접속한다.

```sh
npm run serve --prefix .github/reveal-game-tests
```

`file://` standalone 확인은 생성된 `reveal-game/offline.html`을 Finder에서 두 번 클릭하거나 Playwright smoke test로 수행한다.

## 생성 파일

- `reveal-game/offline.html`: production HTML/CSS/JS를 inline한 deterministic artifact
- `reveal-game/icons/*.png`: 외부 asset 없는 PWA icon
- `fixtures/generated/`: test 실행 때만 만드는 단색 합성 이미지이며 gitignore 대상

실제 행사 이미지나 랭킹 데이터는 repository에 넣지 않는다.
