# changhyun-choi-98.github.io

This repository is for the public GitHub Pages site at <https://changhyunchoi.com>.

It is a Jekyll site based on the [Just the Docs] theme and is deployed with GitHub Pages and GitHub Actions.

## Site URLs

- Primary site URL: <https://changhyunchoi.com>
- GitHub Pages fallback URL: <https://changhyun-choi-98.github.io>
- DNS is managed by Cloudflare.
- The custom domain is configured in GitHub Pages settings.

## Content workflow

The site is page-based, not `_posts`-based. Public articles are normal Markdown pages with `section`, `subcategory`, `date`, `tags`, `summary`, `comments`, and a stable `comment_id`.

`/profile/` contains the imported legacy public academic homepage. The imported profile assets are scoped under `profile/`; do not move `profile/stylesheet.css` into the global Just the Docs theme stylesheet.

Before publishing profile updates, manually review future dates, affiliation text, and company or project-related text.

## Site structure

- Home: `/`
- Profile: `/profile/`
- Work Log: `/work-log/`
- Study: `/study/`
  - Diffusion Tutorial: `/study/diffusion-tutorial/`
  - Algorithm: `/study/algorithm/`
  - System Optimization: `/study/system-optimization/`
  - TensorRT/ONNX Runtime: `/study/tensorrt/`
- Paper: `/paper/`
  - Real-Time Inference: `/paper/real-time-inference/`
- Project: `/project/`
  - Inference Optimization: `/project/inference-optimization/`
    - Shallow-π: `/project/inference-optimization/shallow-pi/`
- Tags: `/tags/` is generated from page tags and hidden from the sidebar.

## Local development

1. Run `bundle install`.
2. Run `bundle exec jekyll serve`.
3. Open <http://127.0.0.1:4000>.

CI uses Ruby 3.3. On this macOS checkout, use the Homebrew Ruby 3.3 toolchain if the system Ruby is selected:

```sh
PATH="/opt/homebrew/opt/ruby@3.3/bin:$PATH" bundle exec jekyll serve
```

[Just the Docs]: https://just-the-docs.github.io/just-the-docs/
