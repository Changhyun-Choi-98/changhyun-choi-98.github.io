---
layout: default
title: Tags
nav_order: 7
permalink: /tags/
---

# Tags

{% assign all_tags = "" | split: "" %}
{% for tagged_page in site.pages %}
  {% if tagged_page.tags %}
    {% assign all_tags = all_tags | concat: tagged_page.tags %}
  {% endif %}
{% endfor %}
{% assign tags = all_tags | uniq | sort %}

{% if tags.size > 0 %}
  {% for tag in tags %}
## {{ tag }}

    {% assign pages_for_tag = site.pages | where_exp: "item", "item.tags contains tag" | sort: "date" | reverse %}
    {% for tagged_page in pages_for_tag %}
- [{{ tagged_page.title }}]({{ tagged_page.url | relative_url }}){% if tagged_page.summary %}: {{ tagged_page.summary }}{% endif %}
    {% endfor %}
  {% endfor %}
{% else %}
No tagged pages yet.
{% endif %}
