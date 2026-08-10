# 이미지 공개 퀴즈 원상 복구 안내

이 절차는 기존 블로그, 맞춤 도메인, DNS, 글을 건드리지 않고 `/reveal-game/` 앱과 전용 CI만 되돌린다. `git reset --hard`, 강제 push, 기록 재작성은 사용하지 않는다.

## 배포 전 확인

1. 필요하면 랭킹 화면에서 JSON 백업을 내보낸다.
2. 최종 배포 보고서의 `DEPLOY_COMMIT`을 확인한다.
3. 작업 트리가 깨끗하고 예상하지 않은 로컬 파일이 없는지 확인한다.

## 복구 명령

```sh
git switch main
git pull --ff-only origin main
git show --stat <DEPLOY_COMMIT>
git revert <DEPLOY_COMMIT>
git diff HEAD^ -- reveal-game .github/reveal-game-tests .github/workflows/reveal-game-ci.yml
git push origin main
```

`<DEPLOY_COMMIT>`에는 배포 후 최종 보고서에 기록된 단일 squash commit hash를 넣는다. 배포 절차에서 squash merge를 강제하므로 한 번의 `git revert`가 앱 전체 변경과 전용 CI를 함께 되돌린다. 자동 보조 스크립트가 commit 또는 push하지 않으며, 각 단계는 사람이 diff를 확인한 뒤 실행한다.

## Pages 확인

1. GitHub Actions에서 Pages workflow 성공을 확인한다.
2. 블로그 홈페이지와 기존 주요 페이지가 정상인지 확인한다.
3. `/reveal-game/`이 제거되었는지 확인한다.
4. 맞춤 도메인, DNS, CNAME, 블로그 글의 예상하지 않은 diff가 없는지 다시 확인한다.

## 설치된 PWA와 브라우저 데이터

- 이미 설치한 PWA에는 캐시된 복사본이 남을 수 있으므로 각 PC에서 PWA를 제거한다.
- 브라우저의 `changhyunchoi.com` 사이트 데이터에서 서비스 워커와 캐시를 지운다.
- 사이트 데이터를 지우면 로컬 랭킹도 삭제되므로 먼저 JSON을 내보낸다.
- 원상 복구 후에도 랭킹을 보존해야 한다면 JSON 파일을 안전하게 별도 보관한다.
