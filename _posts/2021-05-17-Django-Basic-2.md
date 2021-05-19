---
date: 2021-05-17 22:24
title: "Django Basic - Part 2"
categories: DevCourse2 Django
tags: DevCourse2 Django
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

# Model
Django can have ORM (Object Relational DB) and sql.  

## models.py
dir - homepage/models.py:  
```python
class Coffee(models.Model):
    """
    types are
    string: CharField
    number: IntergerField, SmallIntegerField, ...
    logic: BooleanField
    time/date: DateTimeField
    ...
    """
    name = models.CharField(default="",max_length=30) # null=False(default): must have value
    price = models.IntegerField(default=0)
    is_ice = models.BooleanField(default=False)
```  

## admin.py
If register admin in django app, we can manage db in admin page.  

admin.py:  
```python
from django.contrib import admin
from .models import Coffee
# Register your models here.
admin.site.register(Coffee)
```

result:  
<img src="/assets/images/coffee-models.png" width="90%" height="40%">  
Groups, Users are also models as default when creating superuser  

## Migration
To let django know model infomation:  
**makemigration**(notice to django about changes of models) -> **migrate**(save changes)  
```
makemigrations == git add  
migrate == git commit  
```  

1. makemigrations  
    ```
    $ python manage.py makemigrations homepage
    ```  
    result:  
    <img src="/assets/images/makemigration.png" width="90%" height="40%">  

2. migrate  
    ```
    $ python manage.py migrate
    ```  
    result:  
    <img src="/assets/images/migrate.png" width="90%" height="40%">  

    result of page:  
    <img src="/assets/images/migrate-page.png" width="90%" height="40%">  

## Display class's attribute
added two objects in coffee:  
result:  
<img src="/assets/images/two-coffee-obj.png" width="90%" height="40%">  

Display class object as its attribute(name):  
```python
##### models.py #####

class Coffee(models.Model):

    def __str__(self):
        return self.name

    name = models.CharField(default="",max_length=30) # null=False(default): must have value
    price = models.IntegerField(default=0)
    is_ice = models.BooleanField(default=False)
```
result:  
<img src="/assets/images/coffee-object-name.png" width="90%" height="40%">  

## coffe page view
1. Add coffee_view to views.py  

    homepage/views.py:  
    ```python
    from django.shortcuts import HttpResponse, render
    from .models import Coffee # import Coffee class

    ...

    def coffee_view(request):
        coffee_all = Coffee.objects.all() # all, get, filter, ...
        return render(request, 'coffee.html', {"coffee_list":coffee_all})

    ```  

2. Make Coffe Template  

    homepage/template/coffee.html:  
    ```html
    <!DOCTYPE html>
    <html>
        <head>
            <title>Coffee List</title>
            
        </head>

        <body>
            <h1>My Coffee List</h1>
            <p>{%raw%}{{coffee_list}}{%endraw%}</p>
        </body>
    </html>
    ```

3. Add urls.py in webproj/urls.py  

    ```python
    from django.contrib import admin
    from django.urls import path
    from homepage.views import index, coffee_view

    urlpatterns = [
        path('', index), # localhost/
        path('admin/', admin.site.urls), # localhost/admin
        path('coffee/', coffee_view), # localhost/coffee
    ]

    ```  
    result:  
    <img src="/assets/images/coffee-page.png" width="90%" height="40%">  

4. Display as Objects' attributes.  
    ```html
    <!DOCTYPE html>
    <html>
        <head>
            <title>Coffee List</title>
            
        </head>

        <body>
            <h1>My Coffee List</h1>
            {%raw%}{% for coffee in coffee_list %}{%endraw%}
                <p>{%raw%}{{coffee.name}}, {{coffee.price}}{%endraw%}</p>
            {%raw%}{% endfor %}{%endraw%}
        </body>
    </html>
    ```  
    result:  
    <img src="/assets/images/coffee-obj-attributes.png" width="60%" height="40%">  

## Summary
### Process
1. Make a model (ORM) in **homepage/models.py**
2. Register Admin User of the model in **homepage/admin.py**
3. Do Migration (makemigrations(git add), migrate(git commit))
4. Make a template in **template/your.html**
5. Make a view in **homepage/views.py** (input: template-html, data-models class)
6. Add a url in **/urls.py**

# Form
homepage/forms.py:  

## Make forms.py
```python
from django import forms

from .models import Coffee

# forms.ModelForm

class CoffeeForm(forms.ModelForm): # ModelForm을 상속받는 CoffeForm 생성
    class Meta:
        model = Coffee # model
        fields = ('name', 'price', 'is_ice') # name, price, is_ice

```  

## Add to views.py
```python
from django.shortcuts import HttpResponse, render
from .models import Coffee
from .forms import CoffeeForm

...

def coffee_view(request):
    coffee_all = Coffee.objects.all() # all, get, filter, ...
    form = CoffeeForm()
    return render(request, 'coffee.html', {"coffee_list":coffee_all, "coffee_form":form})
```  

## Modify Template
### Without Submit(save) Button
coffee.html:  
```html
    <body>
        <h1>My Coffee List</h1>
        {%raw%}{% for coffee in coffee_list %}{%endraw%}
            <p>{%raw%}{{coffee.name}}, {{coffee.price}}{%endraw%}</p>
        {%raw%}{% endfor%}{%endraw%}

        <form action="">
            {%raw%}{{ coffee_form.as_p}}{%endraw%} <!-- as_p: as paragraph -->
        </form>
    </body>
```  
result:  
<img src="/assets/images/form-noaction.png" width="60%" height="40%">  

### Add Save Button
```html
    <body>
        <h1>My Coffee List</h1>
        {%raw%}{% for coffee in coffee_list %}{%endraw%}
            <p>{%raw%}{{coffee.name}}, {{coffee.price}}{%endraw%}</p>
        {%raw%}{% endfor%}{%endraw%}

        <form action="" method="POST">
            {%raw%}{{ coffee_form.as_p}}{%endraw%} <!-- as_p: as paragraph -->
            <button type="submit">Save</button>
        </form>
    </body>
```  
result:  
<img src="/assets/images/form-save.png" width="60%" height="40%">  


#### Form needs Verification
After push save button:  
<img src="/assets/images/form-error.png" width="100%" height="40%">  

To solve this,  
Add CSRF Token:  
```html
    <body>
        <h1>My Coffee List</h1>
        {%raw%}{% for coffee in coffee_list %}{%endraw%}
            <p>{%raw%}{{coffee.name}}, {{coffee.price}}{%endraw%}</p>
        {%raw%}{% endfor%}{%endraw%}

        <form action="" method="POST">{%raw%}{% csrf_token %}{%endraw%}
            {%raw%}{{ coffee_form.as_p}}{%endraw%} <!-- as_p: as paragraph -->
            <button type="submit">Save</button>
        </form>
    </body>
```  
To get information from the page as "Save" Button,  
Modify views.py.  
```python
def coffee_view(request):
    coffee_all = Coffee.objects.all() # all, get, filter, ...
    # if request is POST
        # Complete Form by POST
        # If valid, save it.
    if request.method=="POST":
        form  = CoffeeForm(request.POST)
        if form.is_valid(): #
            form.save()
    
    form = CoffeeForm()
    return render(request, 'coffee.html', {"coffee_list":coffee_all, "coffee_form":form})
```  
result:  
<img src="/assets/images/form-post.png" width="60%" height="40%">  

