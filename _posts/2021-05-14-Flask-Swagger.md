---
date: 2021-05-14 02:38
title: "Flask API with Swagger"
categories: DevCourse2 Flask
tags: DevCourse2 Flask SQLAlchemy RESTful Swagger
# 목차
toc: True  
toc_sticky: true 
toc_label : "Contents"
---

Flask run as debug mode  
```
$ export FLASK_ENV=development
$ flask run
```  
if not set as dev mode,  
need to keep restart flask like:  
```
$ flask run
```  

# Set up Swagger

calculator example of *cal.py*:  
```python
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property


from flask import Flask
from flask_restplus import Resource, Api, reqparse


# -----------------------------------------------------
# api
# -----------------------------------------------------
app = Flask(__name__)
api = Api(app, version='1.0', title='Calc API',
          description='계산기 REST API 문서',)

ns = api.namespace('calc', description='계산기 API 목록')
app.config.SWAGGER_UI_DOC_EXPANSION = 'list'  # None, list, full


# -----------------------------------------------------
# 덧셈을 위한 API 정의
# -----------------------------------------------------
sum_parser = ns.parser()
sum_parser.add_argument('value1', required=True, help='연산자1')
sum_parser.add_argument('value2', required=True, help='연산자2')


@ns.route('/sum')
@ns.expect(sum_parser)
class FileReport(Resource):
    def get(self):
        """
                Calculate addition
        """
        args = sum_parser.parse_args()

        try:
            val1 = args['value1']
            val2 = args['value2']
        except KeyError:
            return {'result': 'ERROR_PARAMETER'}, 500

        result = {'result': 'ERROR_SUCCESS', 'value': int(val1) + int(val2)}
        return result, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # , debug=True)
```  

To run:  
```
$ python cal.py
```  




Issue:  
AttributeError: 'DummySession' object has no attribute 'query'  


solution:  

edit  ./resources/weapon.py
```python
from db import db
## from 
weapon_schema.load(weapon_json)
## to
weapon_schema.load(weapon_json, session=db.session)
```  

solution ref:  <https://www.gitmemory.com/issue/marshmallow-code/flask-marshmallow/44/508944019>


---  

ref:  
> flask swagger: <https://nurilab.github.io/2020/04/19/we_do_swagger/>  

> flask swagger: <https://minwook-shin.github.io/python-flask-restplus-swagger/>

>  Flask + REST API + Swagger: <https://m.blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221571623994&proxyReferer=https:%2F%2Fwww.google.com%2F>

> Flask, sqlalchemy, marshmallow <https://medium.com/analytics-vidhya/building-rest-apis-using-flask-restplus-sqlalchemy-marshmallow-cff76b202bfb>

