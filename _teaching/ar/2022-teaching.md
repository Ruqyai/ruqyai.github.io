---
title: "دورة هندسة البيانات"
collection: teaching
type: "Course"
permalink: "/ar/teaching/2022-teaching"
venue: "Online"
date: 2022-10-03
location: "Zoom"
---

دورة في هندسة البيانات تركز على كيفية إنشاء لوحة تحكم تحليلية لتويتر في الوقت الفعلي لعرض تغريدات البث المباشر مع ميزة تحليل المشاعر (NLP).



<table class="tg">
    <tr>
        <th class="tg-yw4l"><b>اليوم</b></th>
        <th class="tg-yw4l"><b>العرض التقديمي</b></th>
        <th class="tg-yw4l"><b> الأمثلة </b></th>
    </tr>
    <tr>
        <td class="tg-yw4l"> اليوم الأول </td>
        <td class="tg-yw4l"> <a href="https://bit.ly/DE_Slide1">Slides</a> </td>
        <td class="tg-yw4l"> <a href="https://github.com/Ruqyai/Tuwaiq_Academy/blob/main/Docker-compose.yml"> Run Jupyter Docker image </a> or
            <a href="https://github.com/Ruqyai/Tuwaiq_Academy/blob/main/Jupyter/Colab/PySpark_On_Colab.ipynb">
                <img src="https://colab.research.google.com/assets/colab-badge.svg">
            </a>
        </td>
    </tr>
    <tr>
        <td class="tg-yw4l"> اليوم الثاني </td>
        <td class="tg-yw4l"><a href="https://bit.ly/3M6etal">Slides </a></td>
        <td class="tg-yw4l">
            <a href="https://github.com/Ruqyai/Tuwaiq_Academy/blob/main/Jupyter/Colab/Kafka_on_Colab.ipynb">
                <img src="https://colab.research.google.com/assets/colab-badge.svg">
            </a>
            <a href="https://github.com/Ruqyai/Tuwaiq_Academy/blob/main/Jupyter/Colab/MongoDB_on_Colab.ipynb">
                <img src="https://colab.research.google.com/assets/colab-badge.svg">
                <a href="https://github.com/Ruqyai/Tuwaiq_Academy/blob/main/Jupyter/Colab/Dash_on_colab.ipynb">
                    <img src="https://colab.research.google.com/assets/colab-badge.svg">
                </a>
            </a>
        </td>
    </tr>
    <tr>
        <td class="tg-yw4l"> اليوم الثالث </td>
        <td class="tg-yw4l"><a href="https://bit.ly/3CdN6GL"> Slides </a></td>
        <td class="tg-yw4l"><a href="https://github.com/Ruqyai/Tuwaiq_Academy/blob/main/Docker-compose.yml"> Run the project using Docker </a> or 
        <a href="https://github.com/Ruqyai/Tuwaiq_Academy/blob/main/Jupyter/Colab/Final_Project_on_Colab.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg"></a>
        </td>
    </tr>
</table>

![Dashbord](https://raw.githubusercontent.com/Ruqyai/publish/master/screenshots/dash/9.gif)

## First Day

#### Reivew the [Slides](https://bit.ly/DE_Slide1)   

#### Command to run Just jupyter docker image  

```bash
docker compose up jupyter   
```  

* Copy the token value in the terminal   
* Open the browser    
[http://localhost:10100 ](http://localhost:10100/)  
* Paste it in login field
![Dashbord](https://raw.githubusercontent.com/Ruqyai/publish/master/screenshots/dash/1.jpg)
## Second Day 


#### Reivew the [Slides](https://bit.ly/3M6etal)   

## Third day

#### Reivew the [Slides](https://bit.ly/3CdN6GL)  
![Dashbord](https://raw.githubusercontent.com/Ruqyai/publish/master/screenshots/dash/2.jpg)

#### Commands of docker compose

* build the docker image
```bash
docker compose build  
```  
* run all 

```bash
docker compose up  
```  
* stop and remove all 

```bash
docker compose down 
```  
* clean 
```bash
docker volume rm $(docker volume ls -q)
docker image rm $(docker image ls -q)
docker system prune
```  


<!-- 
Heading 1
======


How to Reduce the Costs of LLMs


## The Presentation

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vQbFKWj5GKIs2kL9peAoXMU93-0QeiSjMa3HcoUMBAhZToy95qo3dutOY5gRbrs7W2s3WAzwy_EBlyU/embed?start=false&loop=false&delayms=3000" frameborder="0" width="100%" height="400px" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>


Heading 2
======


How to Reduce the Costs of LLMs


## The Presentation

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vRDq-Yxg-Ce9t6c4txCcm5Q-EXBhNf7mEyXla0-DRiRt7ivoOo2FexTwzsSGehUnlnT98gcSZAPgz10/embed?start=false&loop=false&delayms=3000" frameborder="0" width="100%" height="400px" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>


Heading 3
======


How to Reduce the Costs of LLMs


## The Presentation

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vRDq-Yxg-Ce9t6c4txCcm5Q-EXBhNf7mEyXla0-DRiRt7ivoOo2FexTwzsSGehUnlnT98gcSZAPgz10/embed?start=false&loop=false&delayms=3000" frameborder="0" width="100%" height="400px" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe> -->

