---
date: 2021-05-27 02:21
title: "Django Chart Visualization"
categories: DevCourse2 Django HTML
tags: DevCourse2 Django HTML 
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

# 1. Add chart.js and jquery (layouts.html)
This is currently located at head, but later I've to move those scripts down to the end of the body tag.  
```html
<!--                 layouts.html                       -->
<!--                 head tag                       -->
...
<!-- Chart js -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.8.0/Chart.min.js" integrity="sha256-Uv9BNBucvCPipKQ2NS9wYpJmi8DTOEfTA/nH2aoJALw=" crossorigin="anonymous"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.8.0/Chart.min.css" integrity="sha256-aa0xaJgmK/X74WM224KMQeNQC2xYKwlAt08oZqjeF0E=" crossorigin="anonymous" />
<!-- jQuery -->
<script src="https://code.jquery.com/jquery-3.3.1.min.js"></script>
...
```  

# 2. Make a naive chart.html template
```html
<!--                  chart.html                          -->
{% raw %}{% extends 'layouts.html' %} {% comment %} load layouts using extends {% endcomment %}

{%block title%}Chart Page{%endblock%}

{% block content %}
{% comment %} {% include 'slider.html'%} {% endcomment %}
<div class="jumbotron bg-white text-black">
  <h1 class="text-center">Chart page</h1><br>
{% comment %} </div>
 <div class="row chartContainer">
  <div class="column"><canvas id="chart1" style="width:30vw; height:50vh"></canvas></div>
  <div class="column"><canvas id="chart2" style="width:30vw; height:50vh"></canvas></div>
</div>  {% endcomment %}
{% endblock %}{% endraw %}
```  

# 3. Upload csv file to Django by bulk_create
## 3-1. Prepare Dataset
csv file with **no header**.  
```
df.to_csv("yourfile.csv", index=False, header=None)
```  
## 3-2. Create a Model 
```python
# EdaApp/models.py
from django.db import models

# Create your models here.
class Product(models.Model):
    InvoiceId = models.CharField(default="", max_length=30) # null=False(default): must have value
    Branch = models.CharField(default="", max_length=2)
    City = models.CharField(default="", max_length=30)
    CustomerType = models.CharField(default="", max_length=6)
    Gender = models.CharField(default="", max_length=6)
    ProductLine = models.CharField(default="", max_length=30)
    UnitPrice = models.FloatField(default=0)
    Quantity = models.IntegerField(default=0)
    Tax = models.FloatField(default=0)
    Total = models.FloatField(default=0)
    Date = models.DateField()
    Time = models.TimeField()
    Payment = models.CharField(default="", max_length=20)
    Cogs = models.FloatField(default=0)
    GrossMargin = models.FloatField(default=0)
    GrossIncome = models.FloatField(default=0)
    Rating = models.FloatField(default=0)
    Year = models.IntegerField(default=0)
    Month = models.IntegerField(default=0)
    Day = models.IntegerField(default=0)
```  

## 3-3. Add data to Model with bulk_create
```
python manage.py shell
```  

```python
>>> import csv
>>> from EdaApp.models import Product
>>> data = open('market.csv')
>>> reader = csv.reader(data)
>>> reader
<_csv.reader object at 0x7f12b88c0f90>
>>> bulk_list = []
>>> for row in reader:
...     bulk_list.append(Product(
...             InvoiceId = row[0],
...             Branch = row[1],
...             City = row[2],
...             CustomerType = row[3],
...             Gender = row[4],
...             ProductLine = row[5],
...             UnitPrice = row[6],
...             Quantity = row[7],
...             Tax = row[8],
...             Total = row[9],
...             Date = row[10],
...             Time = row[11],
...             Payment = row[12],
...             Cogs = row[13],
...             GrossMargin = row[14],
...             GrossIncome = row[15],
...             Rating = row[16],
...             Year = row[17],
...             Month = row[18],
...             Day = row[19]))

>>> Product.objects.bulk_create(bulk_list)
```  

# 4. Add ProductChartView to views.py
template_name is dir of the template to use.  
Using **Sum** in django.db.models, doing groupby **Branchs' sum of UnitPrice**  
For more about annotate, go to appendix.  
```python
from django.db.models import Sum
...
class ProductChartView(TemplateView):
    template_name = "EdaApp/chart.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['qs'] = Product.objects.all()
        context['branch_sum_agg'] = Product.objects.values('Branch').annotate(branch_sum=Sum('UnitPrice'))
        return context
```  
context['branch_sum_agg'] looks like:  
```python
<QuerySet [{'Branch': 'A', 'branch_sum': 18550.8}, {'Branch': 'B', 'branch_sum': 18478.88}, {'Branch': 'C', 'branch_sum': 18567.76}]>
```


# 5. Add the ProductChartView to urls.py
```python
from EdaApp.views import home, eda, ProductChartView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name='home'),
    path('eda/', eda, name='eda'),
    path('chart/', ProductChartView.as_view(), name='chart'),
]
```  

# 6. Making chart in chart.html
Example of Two chart

## 6-1. Add scripts block in layouts.html
Content block is used for h1 tag, so create scripts block to its below.
```html
{%raw%}{% block content%}
{% endblock %}

{% block scripts%}
{% endblock %}{%endraw%}
```  

## 6-2. Insert script in chart.html
Create two charts like this:  
- Go to chart.html  
- loop **branch_sum_agg**
    - get Branch name by x.Branch (labels)
    - get branch sum by x.branch_sum (data)
    - set label as UnitPrice  

```html
{%raw%}{% block scripts%}
<script>
$(document).ready(function(){
  var ctx = document.getElementById('chart1');
  var chart1 = new Chart(ctx, {
      type: 'bar',
      data: {
          labels: [{% for x in branch_sum_agg %}'{{x.Branch}}',{% endfor %}],          
          datasets: [{
              label: 'UnitPrice',
              data: [{% for x in branch_sum_agg %}'{{x.branch_sum}}',{% endfor %}],
              backgroundColor: [
                  'rgba(255, 99, 132, 0.2)',
                  'rgba(54, 162, 235, 0.2)',
                  'rgba(255, 206, 86, 0.2)',
                  'rgba(75, 192, 192, 0.2)',
                  'rgba(153, 102, 255, 0.2)',
                  'rgba(255, 159, 64, 0.2)'
              ],
              borderColor: [
                  'rgba(255, 99, 132, 1)',
                  'rgba(54, 162, 235, 1)',
                  'rgba(255, 206, 86, 1)',
                  'rgba(75, 192, 192, 1)',
                  'rgba(153, 102, 255, 1)',
                  'rgba(255, 159, 64, 1)'
              ],
              borderWidth: 1
          }]
      },
      options: {
          responsive: false,
          scales: {
              y: {
                  beginAtZero: true
              }
          }
      }
  });
  })
</script>
<script>
$(document).ready(function(){
  var ctx = document.getElementById('chart2');
  var chart2 = new Chart(ctx, {
      type: 'bar',
      data: {
          labels: [{% for x in branch_sum_agg %}'{{x.Branch}}',{% endfor %}],
          datasets: [{
              label: 'UnitPrice',
              data: [{% for x in branch_sum_agg %}'{{x.branch_sum}}',{% endfor %}],
              backgroundColor: [
                  'rgba(255, 99, 132, 0.2)',
                  'rgba(54, 162, 235, 0.2)',
                  'rgba(255, 206, 86, 0.2)',
                  'rgba(75, 192, 192, 0.2)',
                  'rgba(153, 102, 255, 0.2)',
                  'rgba(255, 159, 64, 0.2)'
              ],
              borderColor: [
                  'rgba(255, 99, 132, 1)',
                  'rgba(54, 162, 235, 1)',
                  'rgba(255, 206, 86, 1)',
                  'rgba(75, 192, 192, 1)',
                  'rgba(153, 102, 255, 1)',
                  'rgba(255, 159, 64, 1)'
              ],
              borderWidth: 1
          }]
      },
      options: {
          responsive: false,
          scales: {
              y: {
                  beginAtZero: true
              }
          }
      }
  });
  })
</script>
{% endblock %}{%endraw%}
```  

## 6-3. Insert div for chart with two canvases inside of the content block
```html
{%raw%}
{% block content %}
<div class="jumbotron bg-white text-black">
  <h1 class="text-center">Chart page</h1><br>

<div class='chartContainer'>
  <canvas id="chart1" style="width:30vw; height:50vh; display: inline-block; margin-right:200px;"></canvas>
  <canvas id="chart2" style="width:30vw; height:50vh; display: inline-block;"></canvas>
</div>
{% endblock %}{%endraw%}
```  
class='chartContainer' has two canvases.  
To align these charts to the center and horizontally, make style to inline-block.  

<img src='/assets/images/inline-block-chart.png'>  




# Appendix
## &lt;script&gt;

브라우저는 HTML 문서를 처리하다가 &lt;script&gt; 엘리먼트를 만나면 src 속성에 명시된 경로의 파일을 내려받아 자바스크립트 코드를 실행합니다.  
아래와 같이 &lt;script&gt; 엘리먼트가 &lt;body&gt; 엘리먼트의 중간에 오게되면 어떤 일이 일어날까요?

```html
<body>
  <h2>A</h2>
  <script src="./script.js"></script>
  <h2>B</h2>
</body>
```

위 HTML 문서는 브라우저에서 다음과 같이 순차적으로 처리가 됩니다.  
```
    <h2>A</h2>가 화면에 출력됨
    script.js 파일을 내려받아 자바스립트 코드가 실행됨
    <h2>B</h2>가 화면에 출력
```  

전통적으로 HTML 입문자들은 &lt;script&gt; 엘리먼트를 &lt;head&gt; 엘리먼트 안에 넣도록 배우는 경우가 많았습니다. 하지만 실제 프로젝트에서 개발을 하다보면 &lt;body&gt; 엘리먼트의 제일 마지막에서 &lt;script&gt; 엘리먼트를 보게되는 경우가 훨씬 많다는 것을 사용자에게 보다 최적화된 웹페이지 로딩 경험을 제공하려면 &lt;script&gt; 엘리먼트를 &lt;body&gt; 엘리먼트의 마지막에 넣는 것이 유리하기 때문입니다.


## Django annotate Vs. aggregate
```python
>>> from django.db.models import Sum
>>> data = Product.objects.values('Branch').annotate(branch_sum=Sum('UnitPrice'))
>>> data
<QuerySet [{'Branch': 'A', 'branch_sum': 18550.8}, {'Branch': 'B', 'branch_sum': 18478.88}, {'Branch': 'C', 'branch_sum': 18567.76}]>
```  

## References
best for chart.js (2 years ago):  
<https://www.youtube.com/watch?v=1OL5n06kO_w>  

best for django rest api with chart.js, (4 years ago):  
<https://www.youtube.com/watch?v=B4Vmm3yZPgc>  

for multiple charts:  
<https://canvasjs.com/docs/charts/how-to/render-multiple-charts-in-a-page/>  

<br>

ref:  
> <https://testdriven.io/blog/django-charts/>  
> <https://www.freecodecamp.org/news/how-to-create-an-analytics-dashboard-in-django-app/>  
> <https://www.youtube.com/watch?v=jrT6NiM46jk>  
> <https://dowtech.tistory.com/3>  
> <https://m.blog.naver.com/tablesetter/221499332197>  
> <https://www.youtube.com/watch?v=B4Vmm3yZPgc>  
> script: <https://www.daleseo.com/js-script-defer-async/>
> csv to django DB: <https://velog.io/@swhybein/Django-bulkcreate%EC%9C%BC%EB%A1%9C-csv%ED%8C%8C%EC%9D%BC-%EC%98%AC%EB%A6%AC%EA%B8%B0>  
> django-annotate: <https://velog.io/@may_soouu/%EC%9E%A5%EA%B3%A0-Annotate-Aggregate>  
> django-annotate official docs: <https://docs.djangoproject.com/en/3.2/topics/db/aggregation/>  