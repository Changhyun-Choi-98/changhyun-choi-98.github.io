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

<div style="text-align: center; margin: 2.5rem 0 2rem 0;">

<h1 style="font-size: 1.75rem !important; font-weight: 700; margin-bottom: 0.75rem;">
Building robust real-time AI for real-world robots
</h1>

<p style="font-size: 1.05rem; line-height: 1.6; margin: 0 auto; max-width: 720px;">
A research notes archive on <strong>robot intelligence</strong> and <strong>real-time inference systems</strong>
</p>

<style>
.status-note {
  font-size: 0.95rem;
  line-height: 1.6;
  margin: 0.75rem auto 0;
  max-width: 720px;
  font-weight: 600;
  padding: 0.4rem 0.9rem;
  border-radius: 6px;
  display: inline-block;
  color: #a94442;
  background: #fff5f5;
}
@media (prefers-color-scheme: dark) {
  .status-note { color: #e07878; background: rgba(169, 68, 66, 0.18); }
}
[data-theme="dark"] .status-note { color: #e07878; background: rgba(169, 68, 66, 0.18); }
</style>

<p class="status-note">
Currently on a short break between finishing my master's and my next role &mdash; traveling and recharging.
</p>

</div>

<div class="home-brand">
  <img class="home-brand-logo" src="{{ '/assets/images/C_Logo.png' | relative_url }}" alt="C language logo">
  <p class="home-brand-caption">(C for "C"hanghyun "C"hoi . "c"om)</p>
</div>

---

## New Posts

{% assign recent_pages = site.pages | where_exp: "item", "item.date" | sort_pages_for_home %}
{% assign recent_count = 0 %}

{% for recent_page in recent_pages %}
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
