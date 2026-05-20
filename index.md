---
title: Home
layout: default
nav_order: 1
permalink: /
image:
  path: /assets/images/C_Logo.png
  width: 360
  height: 405
  alt: C language logo
---

This site is a technical portfolio and notes archive for building my career in physical AI systems.

<div class="home-brand">
  <img class="home-brand-logo" src="{{ '/assets/images/C_Logo.png' | relative_url }}" alt="C language logo">
  <p class="home-brand-caption">(C for "C"hanghyun "C"hoi . "c"om)</p>
</div>

## New Posts

{% assign recent_pages = site.pages | where_exp: "item", "item.date" | sort: "date" | reverse %}
{% assign recent_count = 0 %}

{% for recent_page in recent_pages %}
  {% if recent_page.path contains "_templates/" %}
    {% continue %}
  {% endif %}
  {% assign recent_count = recent_count | plus: 1 %}
  {% if recent_page.tags contains "Writing" %}
    {% assign recent_status = "Writing" %}
    {% assign recent_status_class = "label-yellow" %}
  {% else %}
    {% assign recent_status = "Complete" %}
    {% assign recent_status_class = "label-green" %}
  {% endif %}
- [{{ recent_page.title }}]({{ recent_page.url | relative_url }}) <span class="home-post-meta">{% if recent_page.tags contains "Korean" %}<span class="label label-blue home-post-label">Korean</span>{% endif %}{% if recent_page.tags contains "English" %} <span class="label label-purple home-post-label">English</span>{% endif %} <span class="label {{ recent_status_class }} home-post-label">{{ recent_status }}</span> <span class="text-grey-dk-000 home-post-date">{{ recent_page.date | date: "%Y-%m-%d" }}</span></span>{% if recent_page.summary %}<br>{{ recent_page.summary }}{% endif %}
  {% if recent_count == 5 %}
    {% break %}
  {% endif %}
{% endfor %}

{% if recent_count == 0 %}
No posts yet.
{% endif %}
