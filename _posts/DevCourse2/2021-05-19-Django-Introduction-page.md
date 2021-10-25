---
date: 2021-05-19 16:07
title: "Django Basic - Create an Introduction Page and Models"
categories: DevCourse2 Django DevCourse2_Django
tags: DevCourse2 Django DevCourse2_Django
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

# Basic Setting
make project:  
```
$ django-admin startproject webproj
```

Run server:  
```
$ cd webproj
$ python manage.py runserver
```

Add App:  
```
$ django-admin startapp homepage
```

add index view in homepage/views.py using render:  
```python
from django.shortcuts import render

# Create your views here.
def index(request):
    myinfo = {}
    myinfo['name'] = 'YILGUK SEO'
    myinfo['major'] = 'Business Administration'
    myinfo['interest'] = 'AI'
    return render(request, 'index.html', {'myinfo':myinfo})
```


Make Template dir in homepage dir (homepage/template)  
Make index.html for home view in template dir.  
get myinfo value from index in views.py  
myinfo here is a dictionary  
To get key, value using for tag,  
use *.items* (in python .items() but not here)  

```html
<!DOCTYPE html>
<html>
    <head>
        <title>Introduction Page</title>
    </head>

    <body>
        <h1>Introduction</h1>
        <p>Hi~ h~ i~</p>
        {%raw%}{% for key, value in myinfo.items %}
        <li>{{ key }} : {{ value }}</li>
        {% endfor %}{%endraw%}
    </body>
</html>
```  

Add index in urls.py  
import and add.  
```python
...
from homepage.views import index

urlpatterns = [
    path('', index), #localhost/
    path('admin/', admin.site.urls), #localhost/admin
]
```

Add installed_apps with app dir name:  
```python
INSTALLED_APPS = [
    ...,
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'homepage',
]
```

Add template dir:  
```python
TEMPLATES = [
    {
        ...,
        'DIRS': [
            os.path.join(BASE_DIR, "homepage","template"),
        ],
        ...,
    ...
    }
```

<img src="/assets/images/intro-home.png" width="50%" height="50%">

# Hide Secret Key
- Find secret key in settings.py  
- make ".env" in proj root dir (in root: webproj/webproj/... .env)
    ```
    # .env
    SECRET_KEY= yoursecretkey
    ```
- import load_dotenv and replace secret key with str in settings.py
    ```python
    ...
    # Protect secret key
    from dotenv import load_dotenv
    load_dotenv()
    ...
    SECRET_KEY = str(os.getenv('SECRET_KEY'))
    ...
    ```
D o n e !


# Apply Static (css, js, bootstrap)
Make static dir with the same level of app dir:  
in webproj like:  
webproj/ homepage/ static/  

Make style.css in static dir:  
style.css 👇
```css
body{
    background-color: aquamarine;
}
```

Apply this css with index.html:  
first use load tag load static  
then add link with href as static tag
```html
{%raw%}{% load static %}
<!DOCTYPE html>
<html>
    <head>
        <link rel="stylesheet" href="{%static 'style.css'%}">
        <title>Introduction Page</title>
    </head>
    <body>
        <h1>Introduction</h1>
        <p>Hi~ h~ i~</p>
        {% for key, value in myinfo.items %}
        <li>{{ key }} : {{ value }}</li>
        {% endfor %}{%endraw%}
    </body>
</html>
```

Go to settings.py:  
make sure 'django.contrib.staticfiles' in INSTALLED_APPS.  
```python
INSTALLED_APPS = [
    ...
    'django.contrib.staticfiles',
    'homepage',
    ...
]
```   
Add STATICFILES_DIRS, STATIC_ROOT like:  
```python
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]

STATIC_ROOT = os.path.join(BASE_DIR, 'assets')
```
정적 파일 접근 경로:  
- 개발 서버: STATICFILES_DIR
- 웹 서버: STATIC_ROOT

static_root를 지정하고 collectstatic 명령어를 실행하면  
정적파일을 한 디렉토리로 수집해서 편리하게 관리가 가능하도록 만들어 준다.  
Run the collectstatic management command:
```
$ python manage.py collectstatic
```  
위 명령어를 실행하면 내가 STATIC_ROOT로 지정한 dir가 생성되면서 기존의 static에 있던 파일들이 copy&paste가 된다.  

result:  
<img src="/assets/images/intro-static.png" width="50%" height="50%">  
D o n e !

# Make App with models(DB)
in webproj dir:  
```
$ python manage.py startapp CoffeShop
```  

CoffeeShop/models.py
```python
class Coffee(models.Model):
    
    def __str__(self):
        return "name: {} ".format(self.name) + "price: {}".format(self.price)
    
    id = models.AutoField(primary_key=True)
    name = models.CharField(default="",max_length=30)
    price = models.IntegerField(default=0)
```

create DB by python shell in app's dir(currently CoffeeShop/):  
```
$ python manage.py shell
```
```
>>> from CoffeeShop.models import Coffee
>>> obj = Coffee(id=1, name="Americano",price=3500)
>>> obj.save()
>>> obj = Coffee(id=2, name="CafeLatte",price=4000)
>>> obj.save()
>>> print(Coffee.objects.all())
<QuerySet [ <Coffee: Coffee object (1)>, <Coffee: Coffee object (2)> ]>
```  

# Register Models
CoffeeShop/admin.py:  
```python
from django.contrib import admin
from CoffeeShop.models import Coffee

# Register your models here.
admin.site.register(Coffee)
```
Register Superuser(root):  
```
$ python manage.py createsuperuser
```  
But got a error.  
```
django.db.utils.OperationalError: no such table: auth_user
```  

Why?  
No table exist since I didn't create any table yet.  
(I intentionally skipped above execution creating DB with python shell)  
Thus, makemigration then migrate.  
Before doing this add the app in settings.py at INSTALLED_APPS.  
Then,  
at webproj/:  
```
$ python manage.py makemigrations CoffeeShop
```
```
$ python manage.py migrate
```
```
$ python manage.py createsuperuser
```  

# Show Coffee list in coffee

### 1. Make a view in views.py
```python
from django.shortcuts import render
from .models import Coffee
# Create your views here.

def coffee_list(request):
    my_coffee_list = Coffee.objects.all()
    return render(request, 'coffee_list.html', {"my_coffee_list":my_coffee_list})
```  

### 2. Make a template for coffee_list
```html
<!DOCTYPE html>
<html>
    <head>
        <title>Coffee List</title>
    </head>

    <body>
        <h1>My Coffee List</h1>
        {%raw%}{% for coffee in my_coffee_list %}
            <ul>{{coffee.name}}: {{coffee.price}}</ul>
        {% endfor%}{%endraw%}
    </body>
</html>
```  

### 3. Add url to urls.py
```python
from django.contrib import admin
from django.urls import path
from homepage.views import index
from CoffeeShop.views import coffee_list

urlpatterns = [
    path('', index), #localhost/
    path('coffee/', coffee_list), #localhost/coffee
    path('admin/', admin.site.urls), #localhost/admin
]
```  

### 4. Add Template path to TEMPLATES in setting.py
```python
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [
            os.path.join(BASE_DIR, "homepage","template"),
            os.path.join(BASE_DIR, "CoffeeShop","template"),
        ],
        ...
]
```  

result:  
<img src="/assets/images/mandatory-coffee-list.png" width="50%" height="50%">  
D o n e !  





# Process

1. Make Template dir and make html template (html file) (at app dir)
2. Add Template dir in setting.py (at root proj)
3. Add a View in views.py as render (at app dir)
4. Import views and add urlpatterns in urls.py(at root proj)



ref:  
> SecretKey: <https://dev.to/vladyslavnua/how-to-protect-your-django-secret-and-oauth-keys-53fl>  
> Static: <https://cupjoo.tistory.com/116>
> official django static: <https://docs.djangoproject.com/ko/3.2/howto/static-files/>