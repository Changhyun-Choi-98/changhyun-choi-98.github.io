---
layout: post
title: "C++ Coding Test Pitfalls"
nav_exclude: true
section: study
subcategory: algorithm
date: 2026-06-08
tags:
  - English
  - C++
  - Writing
language: en
summary: "A concise collection of common C++ mistakes and debugging notes from algorithmic problem-solving practice."
math: true
comments: true
comment_id: "study-algorithm-cpp-pitfalls"
permalink: /study/algorithm/cpp-pitfalls/
---

---

1. Unlike Python, C++ distinguishes between double quotes and single quotes.
    - In C++, `"a"` is not a character literal; it is an ordinary string literal, and its type is `const char[2]`.
2. C++ operator precedence
```cpp
  () [] . ->        // strongest
  ! ~ ++ --         // unary operators
  * / %
  + -
  << >>
  < <= > >=
  == !=
  &
  ^
  |
  &&   // logical AND
  ||   // logical OR
  ?:
  = += -= ...
  ,    // weakest
```
3. Always be mindful of variable scope and type.

4. Convert between `std::string` and `int`.
    - `std::stoi(str)` for `std::string` → `int`
    - `std::to_string(num)` for `int` → `std::string`

{% include comments.html %}
