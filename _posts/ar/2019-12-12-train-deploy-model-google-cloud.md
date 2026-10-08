---
title: "كيف تدرب وتنشر نموذجاً على سحابة قوقل؟"
date: 2019-12-12
permalink: "/ar/posts/2019/12/train-deploy-model-google-cloud/"
excerpt: "بناء حل بتعلم الآلة باستخدام TensorFlow وAI Platform: تدريب موزع على السحابة، ثم نشر النموذج والتنبؤ بأسعار البيوت عبر REST API."
canonical_url: "https://3alam.pro/ru0sa/articles/google"
tags:
  - Google Cloud
  - TensorFlow
  - MLOps
---

> نُشر هذا المقال أولاً في [عالم البرمجة](https://3alam.pro/ru0sa/articles/google).

## Tensorflow + Cloud ML

![خطوات سير العمل العامة لتعلم الآلة على السحابة](/images/posts/train-deploy-model-google-cloud/03.svg)

خطوات سير العمل العامة ل ML على السحابة

## مقدمة

**ماذا ستتعلم؟**

ستتعلم بناء حل بتعلم الآلة يستخدم كلا من Tensorflow and AI Platform للتديب الموزع distributed training على السحابة ومن ثم نشر النموذج على السحابة والتنبؤ باسعار البيوت اونلاين باستخدام REST API و JSON

سنغطي هذه المواضيع:

1.  كيف نستخدم Tensorflow's high level Estimator API
2.  كيف ننشر tensorflow كود للتدريب الموزع distributed training في السحابة cloud
3.  كيف نقيم النتائج باستخدام TensorBoard
4.  كيف ننشر النموذج الناتج the resulting model في السحابة cloud للتنبؤ اونلاين

### ماهو tf.estimator ؟

واجهة برمجة تطبيقات **TensorFlow** عالية المستوى تعمل على تبسيط برمجة التعلم الآلي. وتساعد على:

- تدريب training
- تقييم evaluation
- تنبؤ prediction
- تصدير للخدمة export for serving

## لنبدأ

افتح**[Google Console](https://console.cloud.google.com/)** سنفترض أنك مسجل من قبل وجاهز لإستخدامها

افتح **Cloud Shell** ومن ثم اضعط علي زر الاستمرار

![](/images/posts/train-deploy-model-google-cloud/04.png)

![](/images/posts/train-deploy-model-google-cloud/05.png)

ستفتح لك نافذة تشبة هذه

![](/images/posts/train-deploy-model-google-cloud/06.png)

الان قم بانشاء **Storage Bucket** للتخزين بإتباع الخطوات التالية:

- في **GCP Console** انقر على  **Navigation menu** ومن تم اختر **Storage**
- انقر على **Create bucket**
- اختر اسم فريد غير مكرر واختر المنطقة المناسبة لك او اجعلها متعددة
- انقر **Create**
-  

## تشغيل AI Platform

في **GCP Console** انقر على  **Navigation menu** ومن تم اختر **AI Platform** ثم **Notebooks**

![](/images/posts/train-deploy-model-google-cloud/07.png)

![](/images/posts/train-deploy-model-google-cloud/08.png)

اختر **New Instance**  

ثم اختر **Tensorflow Enterprise 1.XX**  ومن ثم **Without GPU**

![](/images/posts/train-deploy-model-google-cloud/09.png)

![](/images/posts/train-deploy-model-google-cloud/10.png)

*سيستغرق الأمر من ٢ الى ٣ دقائق*

*ثم انقر فوق فتح* **JupyterLab** *سيتم فتح نافذة* **JupyterLab** *في علامة تبويب جديدة.*

![](/images/posts/train-deploy-model-google-cloud/11.png)

الخطوة التالية تحميل ال **Notebook**  
اختر **Terminal** كما هو موضح بالصورة

![](/images/posts/train-deploy-model-google-cloud/12.png)

## استنساخ الكود clone

في موجه الأوامر ، اكتب الأمر التالي (اختر إما الامر بالخيار الأول أو الخيار الثاني) ثم اضغط على Enter.

الخيار الأول: هنا الشرح مختصر

```bash
git clone https://github.com/Ruqyai/Deploy-a-Model-on-Cloud-ML-Engine.git
```

الخيار الثاني: هنا في اسهاب بالشرح

```bash
git clone https://github.com/vijaykyr/tensorflow_teaching_examples.git
```

**فتح وتنفيذ ال Notebook**

من **Jupyter console**  

للخيار الأول: اختر  
**\> cloud-ml-housing-prices.ipynb**

للخيار الثاني: اذهب الى واختر  
**tensorflow_teaching_examples \> housing_prices \> housing_prices \> cloud-ml-housing-prices.ipynb**

الآن سيظهر لك ال Notebook ابدآ بقراءة الأكواد وتشغيلها

### ترجمة من مختبر على [Qwiklabs](https://www.qwiklabs.com/focuses/3644?parent=catalog)
