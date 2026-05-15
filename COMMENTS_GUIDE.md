# Comments Guide

This site uses Disqus for public page comments.

## Site Configuration

The Disqus shortname is configured in `_config.yml`:

```yaml
comments:
  provider: "disqus"
  enabled: true
  disqus:
    shortname: "changhyun-choi"
```

Use the real Disqus shortname after creating the site in Disqus. If `_config.yml` still contains `CHANGE_ME_DISQUS_SHORTNAME`, replace it before publishing comments. While the placeholder is present, the site shows a setup warning on posts with comments enabled and does not load the Disqus embed script.

Do not commit JavaScript secrets, private keys, or private credentials. Disqus comments use the public shortname only.

## Per-Post Comments

Enable comments only on real public articles:

```yaml
comments: true
comment_id: "section-subcategory-stable-slug"
```

Add the comments include at the bottom of the article body:

```liquid
{% include comments.html %}
```

`comment_id` is the stable Disqus thread identifier. Choose it once and keep it unchanged even if the page URL changes. If `comment_id` is missing, the comments include falls back to `page.url`, but explicit stable IDs are preferred.

Do not enable comments on private or sensitive pages. Do not enable comments on category landing pages, tag pages, the home page, or the imported legacy profile page.

## Disqus Admin Setup

1. Create a site in the Disqus admin console.
2. Confirm the site's shortname.
3. Put the real shortname in `_config.yml`.
4. In Disqus Reactions settings, create these six reactions:
   - Upvote
   - Funny
   - Love
   - Surprised
   - Angry
   - Sad
5. Prefer disabling guest commenting so only logged-in users can comment.
6. Configure the moderation policy.
7. Enable spam filtering and configure blocked words.
