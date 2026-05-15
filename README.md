# changhyun-choi-98.github.io

This repository is for the public GitHub Pages site at <https://changhyunchoi.com>.

It is a Jekyll site based on the [Just the Docs] theme and is deployed with GitHub Pages and GitHub Actions.

## Site URLs

- Primary site URL: <https://changhyunchoi.com>
- GitHub Pages fallback URL: <https://changhyun-choi-98.github.io>
- DNS is managed by Cloudflare.
- The custom domain is configured in GitHub Pages settings.

## Content workflow

See [CONTENT_GUIDE.md](CONTENT_GUIDE.md) before adding Work Log, Study, Paper, or Project posts.

See [COMMENTS_GUIDE.md](COMMENTS_GUIDE.md) before enabling comments or changing the Disqus configuration.

`/profile/` contains the imported legacy public academic homepage. The imported profile assets are scoped under `profile/`; do not move `profile/stylesheet.css` into the global Just the Docs theme stylesheet.

Before publishing profile updates, manually review future dates, affiliation text, and company or project-related text.

## Local development

1. Run `bundle install`.
2. Run `bundle exec jekyll serve`.
3. Open <http://127.0.0.1:4000>.

[Just the Docs]: https://just-the-docs.github.io/just-the-docs/
