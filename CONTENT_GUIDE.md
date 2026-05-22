# Content Guide

This site is a public technical portfolio for notes on robotics AI, AI systems performance, GPU optimization, research, and public projects.

## Current Content Map

The site uses normal Jekyll pages and Just the Docs navigation. It does not use `_posts` or Jekyll collections for articles.

Current sidebar tree:

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
    - Further Optimizing Shallow-π: `/project/inference-optimization/further-optimizing-shallow-pi/`
- Tags: `/tags/` is generated from page tags and intentionally hidden from the sidebar.

Current list-backed article areas:

- Work Log: `section: work-log`; no published article pages yet.
- Diffusion Tutorial: `section: study`, `subcategory: diffusion-tutorial`; currently includes `study/diffusion-tutorial/0_VAE.md`.
- Algorithm: `section: study`, `subcategory: algorithm`; landing page exists, no article pages yet.
- System Optimization: `section: study`, `subcategory: system-optimization`; landing page exists, no article pages yet.
- TensorRT/ONNX Runtime: `section: study`, `subcategory: tensorrt`; landing page exists, no article pages yet.
- Real-Time Inference: `section: paper`, `subcategory: real-time-inference`; currently includes OxyGen, DEFLECT, and Realtime-VLA FLASH paper notes.
- Further Optimizing Shallow-π: `section: project`, `subcategory: further-optimizing-shallow-pi`; currently includes implementation and profiling notes.

`/project/inference-optimization/` is a topic parent, not a post list. Do not use `subcategory: inference-optimization` for an article unless a matching `content-list.html` landing page is added.

## Writing Style

Most public articles are Korean technical notes with English technical terms left in English when that is clearer. Keep the tone direct and study-log-like: explain the problem, the mechanism, the measured result or open question, and why it matters.

Use front matter consistently:

- `summary` should be one concise Korean sentence.
- `tags` should include `Korean` or `English`; add `Writing` while the article is still in draft or incomplete state.
- The Home page treats pages without `Writing` as `Complete`.
- Use `math: true` only when MathJax is needed.
- Use `comments: true`, a stable `comment_id`, and `{% include comments.html %}` only for real public article pages.

Common body patterns:

- Paper notes start with a title and a `<aside class="series-preface" markdown="1">` block containing public metadata such as authors, affiliations, links, and submitted or updated dates. They then explain the core idea, method, experiments, limitations, and personal questions as needed.
- Project notes start with a public environment/context preface when measurements are involved. Shared-server caveats should stay explicit when latency or training numbers are shown.
- Study notes can use a short preface to explain the series context and the learning goal.
- Figures use Markdown image syntax followed by an italic caption and `{: .figure-caption}`.
- Long code or formulas can be placed in `<details markdown="1">` blocks when they would interrupt the main reading flow.

## Adding a New Post

1. Copy one file from `_templates/`.
2. Place the copied file under the matching list-backed section:
   - Work log: `work-log/YYYY-MM-DD-slug.md`
   - Study note: `study/<subcategory>/YYYY-MM-DD-slug.md`
   - Paper note: `paper/<subcategory>/YYYY-MM-DD-slug.md`
   - Project note: `project/<topic>/<leaf-subcategory>/YYYY-MM-DD-slug.md`
3. Replace all placeholders such as `TITLE_HERE`, `YYYY-MM-DD`, `SLUG_HERE`, `SUBCATEGORY_HERE`, `PUBLIC_SAFE_SUMMARY_HERE`, and metadata placeholders.
4. Keep `nav_exclude: true` for individual posts unless there is a specific reason to show the post in the sidebar.
5. Run `bundle exec jekyll build` before publishing.

If a new Study, Paper, or Project leaf category is needed, add its landing page first and include `_includes/content-list.html` with the exact `section` and `subcategory` values that article pages will use.

## Front Matter Rules

Use `section` to group the post into a top-level content area:

- `work-log`
- `study`
- `paper`
- `project`

Use `subcategory` to match the folder and landing page where the post should appear. Examples:

- `diffusion-tutorial`
- `tensorrt`
- `algorithm`
- `system-optimization`
- `real-time-inference`
- `further-optimizing-shallow-pi`

Use category landing pages to preserve the Just the Docs sidebar hierarchy. Individual posts should usually omit `parent` and `grand_parent` so they appear in landing-page lists without creating sidebar children:

- Work log post: no `parent` or `grand_parent`; it appears from the Work Log landing page list.
- Study, Paper, and Project posts: use `section` and `subcategory` to choose the landing-page list.

Use a section-specific permalink:

- `/work-log/SLUG_HERE/`
- `/study/<subcategory>/SLUG_HERE/`
- `/paper/<subcategory>/SLUG_HERE/`
- `/project/<subcategory>/SLUG_HERE/`

## Comments

Real public articles may use Disqus comments:

```yaml
comments: true
comment_id: "section-subcategory-stable-slug"
```

Do not use comments on category landing pages, subcategory index pages, tag pages, the home page, or the imported legacy profile page.

Set `comment_id` once and do not change it. Disqus uses this value as the stable thread identifier, so changing a page URL can split comment threads unless the original `comment_id` is preserved.

## Navigation and Lists

Individual posts should use `nav_exclude: true` by default. This keeps the Just the Docs sidebar focused on top-level sections and subcategory landing pages.

Work Log uses `_includes/content-list.html` to show posts with `section: work-log`. Subcategory landing pages for Study, Paper, and Project use the same include with matching `section` and `subcategory` values. Posts excluded from the sidebar still appear in these lists and on the Tags page.

The `_templates/` directory is not a Jekyll collection. Template files are copy sources for authors and should not be rendered as public pages.

## Supporting Files

These files are active site hooks or authoring helpers:

- `_includes/head_custom.html`: favicons, theme-mode bootstrap script, and MathJax loading when `page.math` is true.
- `_includes/header_custom.html`: theme switcher controls.
- `_includes/nav_footer_custom.html`: sidebar social links and Just the Docs credit.
- `_includes/footer_custom.html` and `_includes/post-navigation.html`: previous/list/next navigation for individual article pages.
- `_includes/content-list.html`: section/subcategory article lists with client-side pagination.
- `_includes/comments.html`: Disqus embed guard and per-page Disqus configuration.
- `_includes/js/custom.js`: appended into the Just the Docs JavaScript bundle; preserves expanded sidebar state.
- `_sass/custom/setup.scss`: local Just the Docs color variables.
- `_sass/custom/custom.scss`: local styles for figures, prefaces, Home labels, theme switcher, social links, comments, post navigation, and pagination.
- `_sass/color_schemes/dark.scss`: local dark color scheme.
- `_templates/*.md`: copy sources for new article pages.

Do not remove these only because they have no explicit `{% include %}` reference in a Markdown file. Several are loaded by Just the Docs naming conventions.

## Public-Safe Rules

Do not commit or publish:

- Company-internal code.
- Internal benchmark numbers.
- Internal architecture, pipeline, or deployment structure.
- Internal screenshots, logs, traces, or profiler output.
- Unreleased project names.

Allowed sources include:

- Public papers.
- Public documentation.
- Personal toy examples.
- Reproductions based on public repositories.

Company-related experience must be abstracted and stripped of sensitive implementation details before publication.
