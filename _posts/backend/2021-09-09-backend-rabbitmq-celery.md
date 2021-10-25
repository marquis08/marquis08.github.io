---
date: 2021-09-08 20:10
title: "Backend using RabbitMQ and Celery with docker"
categories: DevCourse2 Docker Backend RabbitMQ Celery
tags: DevCourse2 Docker Backend RabbitMQ Celery
# 목차
toc: true  
toc_sticky: true 
toc_label : "Contents"
---

Build web api with fast api framework on top of python with Celery distributed queue and RabbitMQ.  

At some points, you need to call micro services from the outside.    

For example, from web browser or from another client, you need to have entrypoint and you need to have this web api.  

And when request comes from web api, it may time to process that request in micro services especially if you talk about ML, it may take sometime like if you train the model for example.   

So, we should be able to cbuild asynchronous web api and client should be able to place as many requests as he wants, and get back the reuslt when the result is ready.  
So there should be two endpoints available.  
The first one where we initiate the task and the second one where we get a response and check the status of the task.  

Using FastAPI is to implement a rest endpoint.  
Using Celery is to implement asynchronous tasks.  
So when the request arrives through FastAPI we create a synchronous Celery task which is placed in a queue and to transfer the background, not blocking the main request.  
When a job is done, the task returns the result and a fast api endpoint where we expose the result of the task from there we can check the status of the task and get back the result.  

When a Celery task is running, it needs to send the event to group of microservices to do the job.  

We transmit the event using RabbitMQ remote procedure call(RPC).
And  whatever a microservice is supposed to handle the task, a microservice gets the message, a microservice do the job and returns back the result.  
And when celery task gets back the result, it propagates the task back to FastAPI endpoint where the result becomes available.

# Example
## Katana-skipper - Engine
### Instructions
- start FastAPI
  - `uvicorn endpoint:app --reload`
- start Celery queue
  - `celery -A api.worker worker --loglevel=INFO`
- start test client for RabbitMQ (testing)
  - `python event_receiver_test.py`
- 



# Appendix
- Remote Procedue Call
- Task
- Celery
- RabbitMQ
- 

> Web API with FastAPI, RabbitMQ and Celery: <https://www.youtube.com/watch?v=a0ODIWsCgDI&ab_channel=AndrejBaranovskij>  
> Serving ML Model with Docker, RabbitMQ, FastAPI and Nginx: <https://www.youtube.com/watch?v=vFoRP6ztcrs>  
> Event-Driven Microservice with RabbitMQ and FastAPI: <https://www.youtube.com/watch?v=syRmaDVv59k&ab_channel=AndrejBaranovskij>