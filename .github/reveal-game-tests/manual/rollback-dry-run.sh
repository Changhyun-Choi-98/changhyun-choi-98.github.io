#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
  echo "사용법: sh rollback-dry-run.sh <DEPLOY_COMMIT>"
  exit 2
fi

deploy_commit=$1
git cat-file -e "${deploy_commit}^{commit}"
set -- $(git rev-list --parents -n 1 "$deploy_commit")
if [ "$#" -ne 2 ]; then
  echo "중단: DEPLOY_COMMIT은 부모가 하나인 squash commit이어야 합니다."
  exit 1
fi
echo "검토할 배포 commit: ${deploy_commit}"
git show --stat --oneline "$deploy_commit"
echo "되돌릴 경로 diff 미리보기:"
git diff "${deploy_commit}^" "$deploy_commit" -- reveal-game .github/reveal-game-tests .github/workflows/reveal-game-ci.yml
echo "이 script는 checkout, revert, commit, push를 수행하지 않았습니다."
