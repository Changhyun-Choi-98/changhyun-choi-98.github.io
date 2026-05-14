# Content Guide

This site is a public technical portfolio for notes on robotics AI, AI systems performance, GPU optimization, research, and public projects.

## Adding a New Post

1. Copy one file from `_templates/`.
2. Place the copied file under the matching section:
   - Work log: `work-log/YYYY-MM-DD-slug.md`
   - Study note: `study/<subcategory>/YYYY-MM-DD-slug.md`
   - Paper note: `paper/<subcategory>/YYYY-MM-DD-slug.md`
   - Project note: `project/<subcategory>/YYYY-MM-DD-slug.md`
3. Replace `TITLE_HERE`, `YYYY-MM-DD`, `SLUG_HERE`, `summary`, and `tags`.
4. Keep `nav_exclude: true` for individual posts unless there is a specific reason to show the post in the sidebar.
5. Run `bundle exec jekyll build` before publishing.

## Front Matter Rules

Use `section` to group the post into a top-level content area:

- `work-log`
- `study`
- `paper`
- `project`

Use `subcategory` to match the folder and landing page where the post should appear. Examples:

- `tensorrt`
- `algorithm`
- `gpu-acceleration`
- `language-c`
- `language-cpp`
- `robotic-foundation-model`
- `real-time-inference`
- `paper-reproduction`
- `inference-optimization`

Use category landing pages to preserve the Just the Docs sidebar hierarchy. Individual posts should usually omit `parent` and `grand_parent` so they appear in landing-page lists without creating sidebar children:

- Work log post: no `parent` or `grand_parent`; it appears from the Work Log landing page list.
- Study, Paper, and Project posts: use `section` and `subcategory` to choose the landing-page list.

Use a section-specific permalink:

- `/work-log/SLUG_HERE/`
- `/study/<subcategory>/SLUG_HERE/`
- `/paper/<subcategory>/SLUG_HERE/`
- `/project/<subcategory>/SLUG_HERE/`

## Navigation and Lists

Individual posts should use `nav_exclude: true` by default. This keeps the Just the Docs sidebar focused on top-level sections and subcategory landing pages.

Work Log uses `_includes/content-list.html` to show posts with `section: work-log`. Subcategory landing pages for Study, Paper, and Project use the same include with matching `section` and `subcategory` values. Posts excluded from the sidebar still appear in these lists and on the Tags page.

The `_templates/` directory is not a Jekyll collection. Template files are copy sources for authors and should not be rendered as public pages.

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
