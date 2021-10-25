---
title: "DevCourse2_Django"
layout: archive
permalink: categories/DevCourse2_Django
author_profile: true
sidebar_main: true
---


{% assign posts = site.categories.DevCourse2_Django %}
{% for post in posts %} {% include archive-single.html type=page.entries_layout %} {% endfor %}