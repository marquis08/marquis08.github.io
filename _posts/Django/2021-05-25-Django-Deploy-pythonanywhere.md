---
date: 2021-05-25 00:21
title: "Django Deploy - pythonanywhere"
categories: DevCourse2 Django HTML Deploy
tags: DevCourse2 Django HTML Deploy
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

# Deploy Django Wep app - Pyhonanywhere

There are many references.  

## 1. Make Django App
While making Django app, make **DEBUG = True**.  
Happy Coding ~ ~ 👍

## 2. Push to Github
Caution:  
- DEBUG = False  
- ALLOWED_HOST = ['*'] -> This makes me crazy.
- SECRETKEY 
    - Hiding is impossible since pythonanywhere is based on git repo.
    - Even though make key.txt or key.json invisible by .gitignore, pythonanywhere can get this.  

## 3. What made me confusing is directory.
I initialized git repo to parent directory.  
What I did:  
- MyProj
    - Main
        - assets
        - edapage
        - Main
        - static
        - template
    - .gitignore
    - README.md

What normally does:  
- Main
    - assets
    - edapage
    - Main
    - static
    - template
    - .gitignore
    - README.md

## 4. Pythonanywhere
### 4-1. Add a new web app
Go to Web menu  
**Add a new web app**  
Select  
**Manual Configuration** (since we already made django project.)  
### 4-2. Bash
Go to Console menu  
Click **Bash** to install django project and venv.  
```
$ git clone https://github.com/username/yourproj.git
$ cd proj (to manage.py)
$ virtualenv --python=python3.8 myvenv (myvenv is venv name, this is venv dir)
$ source myvenv/bin/activate
$ pip install django
$ python manage.py migrate
```  
### 4-3. Setting Web
#### Code menu
- modify Source code:  
    ```
    /home/yilgukseo/MonthlyEDA/Main(manage.py)
    ```
- modify WSGI config file:
    ```python
    import os
    import sys

    path = '/home/yilgukseo/MonthlyEDA/Main' # manage.py
    if path not in sys.path:
        sys.path.append(path)

    os.environ['DJANGO_SETTINGS_MODULE'] = 'Main.settings' #settings.py

    from django.core.wsgi import get_wsgi_application
    from django.contrib.staticfiles.handlers import StaticFilesHandler
    application = StaticFilesHandler(get_wsgi_application())
    ```  
#### Virtualenv menu
I created venv at Main(manage.py) dir, so venv dir is like this:  
```
/home/yilgukseo/MonthlyEDA/Main/myvenv 
```
#### Reload
reload
#### Go to the site
[yilgukseo.pythonanywhere.com](yilgukseo.pythonanywhere.com)  


ref:  
> myteammate 👍: <https://taksw222.tistory.com/64?category=481008>  
> <https://windybay.net/post/2/>  
> <https://morningbird.tistory.com/24>  
