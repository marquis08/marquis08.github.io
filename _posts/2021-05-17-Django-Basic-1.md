---
date: 2021-05-17 17:36
title: "Django - Part 1"
categories: DevCourse2 Django
tags: DevCourse2 Django
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

# MVT Pattern
<img src="/assets/images/django-mvt.png" width="90%" height="40%">  



# Installation
## Vertual Environment (conda environment)
Create conda env with name of django  
```
$ conda create -n django
```  
## Install Django
Enter django env by:  
```
$ conda activate django
```  

install pip:  
```
$ conda install pip
```

install django using pip:  
```
$ pip install django
```

# Make Django Project
make project:  
```
$ django-admin startproject webproj
```
Run server:  
```
$ python manage.py runserver
```
Success:  
<img src="/assets/images/django-runserver.png" width="90%" height="40%">  

# Make Django App
go to webproj dir:  
db.qlite3, manage.py, webproj

```
$ django-admin startapp homepage
```
now:  
db.qlite3, homepage, manage.py, webproj  

# View
dir - homepage/views.py:  
```python
from django.shortcuts import HttpResponse, render

# Create your views here.

def index(request):
    return HttpResponse("Hello World!")
```

dir - webproj/urls.py:  
```python
from django.contrib import admin
from django.urls import path
from homepage.views import index

urlpatterns = [
    path('', index),
    path('admin/', admin.site.urls),
]
```

dir - webproj/setting.py:  
```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'homepage',
]
```

## Admin Registration
dir - django-proj/webproj
Migrate first:  
```
$ python manage.py migrate
```  
result:  
<img src="/assets/images/run-migrate.png" width="60%" height="60%">  

Resigter Admin account:  
```
$ python manage.py createsuperuser
```  
login as admin:  
<img src="/assets/images/admin-page.png" width="100%" height="60%">  

# Template
## h1 header
dir - homepage/views.py:  
```python
def index(request):
    return HttpResponse("<h1>Hello World!</h1>")
```  
result:  
<img src="/assets/images/h1-header.png" width="100%" height="60%">  

## Render
**render**:  
render(request, '~.html', context)
이런 형식으로 render를 사용해서 template에 context를 채워넣어 표현한 결과를 HttpResponse 객체와 함께 return하는 함수다.

1. make the **html file** as request in homepage/template dir
2. input the html file to **render** input in homepage/views.py
3. add template path to **TEMPLATES**'s 'DIRS' variable in webproj/settings.py (using os module)

### 1. Make html file in homepage/template
dir - homepage/template/index.html:  
```html
<!DOCTYPE html>
<html>
    <head>
        <title>Python django example</title>
        
    </head>

    <body>
        <h1>Title</h1>
        <p>blah</p>
    </body>
</html>
```

### 2. Input html file to **render**
```python
def index(request):
    # return HttpResponse("<h1>Hello World!</h1>")
    return render(request, 'index.html', {})
```

### 3. Add template path 
to **TEMPLATES**'s 'DIRS' variable in webproj/settings.py  
(using os module)  
webproj/setting.py:  
```python
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            os.path.join(BASE_DIR, "homepage","template")
            ],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
```  

### Django-template language in html
dir - homepage/views.py  
input as dictionary
```python
def index(request):
    num = 10
    return render(request, 'index.html', {"my_num":num})
```  
Using double curly bracket.  
get value by dict key:  
```html
<!DOCTYPE html>
<html>
    <head>
        <title>Python django example</title>
        
    </head>

    <body>
        <h1>Title</h1>
        <p>blah</p>
        <p> {% raw %}{{ my_num }}{% endraw %} </p>
    </body>
</html>
```  
result:  
<img src="/assets/images/django-template-language.png" width="100%" height="60%">  

### Template Filter
ex: length, upper, ...  
1. Modify homepage/views.py
2. Modify homepage/tempate/index.html

```python
def index(request):
    name = "John"
    return render(request, 'index.html', {"my_name":name})
```
```html
...
    <body>
        <h1>Title</h1>
        <p>blah</p>
        <p>{%raw%}{{ my_name|length }}{%endraw%}</p>
    </body>
...
```  
result:  
<img src="/assets/images/template-filter.png" width="60%" height="60%">  

### Template Tag
```html
...
    <body>
        <h1>Title</h1>
        <p>blah</p>
        <p>{%raw%}{{ my_name|length }}{%endraw%}</p>
        {%raw%}{% tag ... %}{%endraw%}
        {%raw%}{% endtag ... %}{%endraw%}
    </body>
...  

```  
#### for tag
for tag:  
```html
...
    <body>
        <h1>Title</h1>
        <p>blah</p>
        {%raw%} {% for elem in my_lst %} {%endraw%}
            <p>{%raw%}{{elem}}{%endraw%}</p>
        {%raw%} {% endfor %} {%endraw%}
    </body>
...
```  
result:  
<img src="/assets/images/for-tag.png" width="60%" height="60%">  

#### if tag (if not)
if tag exercise return odd elements:  
```html
...
    <body>
        <h1>Title</h1>
        <p>blah</p>
        {%raw%} {% for elem in my_lst %} {%endraw%}
            {%raw%} {% if not elem|divisibleby:"2" %} {%endraw%}
                <p>{%raw%}{{elem}}{%endraw%}</p>
            {%raw%} {% endif %} {%endraw%}
        {%raw%} {% endfor %} {%endraw%}
    </body>
...
```
- if and if not is the same as python.  
- Must close the tag
- using filter with variable(in this case elem)




ref:  
>django-mvt image: <https://butter-shower.tistory.com/49>  
>django render official: <https://docs.djangoproject.com/ko/1.11/intro/tutorial03/>  