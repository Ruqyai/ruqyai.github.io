---
title: "خارطة طريق التعلم العميق (Deep Learning)"
date: 2020-09-12
permalink: "/ar/posts/2020/09/deep-learning-roadmap/"
excerpt: "محاولة لرسم خارطة طريق للتعلم العميق: الأسس التي يتفرع منها كل شيء، والفروع الرئيسية، والمناطق التي ما زالت غامضة."
canonical_url: "https://fihm.ai/tutorials/deep-learning-roadmap/"
tags:
  - Deep Learning
  - Roadmap
---

> نُشر هذا المقال أولاً في [منصة فهم](https://fihm.ai/tutorials/deep-learning-roadmap/) بتاريخ 12 سبتمبر 2020.

![](/images/posts/deep-learning-roadmap/01.jpg)

## محاولة رسم خارطة طريق التعلم العميق  \[1\]

بعد بضع سنوات من تتبع تطورات التعلم العميق ، لم يكلف أحد عناء وضع خريطة لما يجري! لذلك قرر الكاتب الخروج سريعا بخارطة طريق للتعلم العميق. وهذه مجرد خارطة جزئية ولا تغطي آخر التطورات. وكما يتضح للملم بهذا العلم أنها ليست شاملة ولا تغطي العديد من الأفكار فهناك الكثير من الأفكار الأخرى التي لم يتم إدراجها أو إعادة ترتيبها بالخارطة. على أي حال ، **هذه بداية  ونأمل أن يبدأ الأشخاص في التوسع أكثر في ذلك.**

تبدأ الخارطة من وجهة نظر الكاتب بهذه الأسس  وتعتبر المستوى الأعلى الذي يتفرع منه البقية :  

![](/images/posts/deep-learning-roadmap/02.jpg)

ثم يكمل تصوره للخارطة بهذه الفروع:  

![](/images/posts/deep-learning-roadmap/03.jpg)

وأشار الكاتب أن التعلم غير الموجه  هو “المنطقة المظلمة” حيث نحتاج إلى مزيد من الوضوح حتى نتمكن من رسم الخارطة بشكل أفضل.  

ويقول لا يزال هناك الكثير مما يتعين القيام به ولايزال هناك العديد من التفاصيل التي لم يتم تغطيتها ونحن في المراحل الأولى من تطور التعلم العميق :  

![](/images/posts/deep-learning-roadmap/04.jpg)

راجع [هذا](https://medium.com/intuitionmachine/five-levels-of-capability-of-deep-learning-ai-4ac1d4a9f2be)  

ونصح الكاتب باقتناء كتاب THE DEEP LEARNING AI PLAYBOOK إذا كنت تعتقد أنك تحتاج لمزيد من التوضيح حول التعليم العميق .

![](/images/posts/deep-learning-roadmap/05.png)

مزيد من التغطية حول الكتاب [هنا](https://deeplearningplaybook.com/)

هذا بشكل عام لمن يرغب بمعرفة المسارات المتعددة للتعلم العميق وصلة بعضها ببعض. ماذا بالنسبة لمن يرغب بالتعلم من المبتدئين؟!.. **كيف يبدأ ومن أين يبدأ؟**

## خارطة طريق لتعلم التعلم العميق للمبتدئين \[2\]

كما يتضح لنا معظم محتوى التعلم العميق على الإنترنت هو:

–  إما على مستوى متقدم جدًا حيث يتحدثون على الفور عن أحدث الأبحاث دون إعطاء أي تفاصيل عن التنفيذ ويكون فقط موجه نحو المستخدمين المتقدمين و يتجاهلون تماما المبتدئين.

 –  أو على مستوى المبتدئين حيث يتحدثون فقط عن الوحدات عالية المستوى high level modules  في مكتبات DL (على سبيل المثال ، Keras of Tensorflow 2.0. فهناك عدد لا يحصى من مقاطع فيديو YouTube التي تطلق على نفسها “Tensorflow2.0 Tutorials” وتغطي فقط Keras )

ويقول الكاتب نادرًا ما صادفت شيئًا يستحق الإشادة ولا يقع في التصنيف المذكورة أعلاه. فيما يلي خارطة طريق أتمنى لو حصلت عليها عندما بدأت رحلتي إلى التعلم العميق. يمكن اتباعها إذا كان لديك تجربة وأساسيات بايثون ، ولديك الصبر والدافع لإكمال الطريق . هذه النقاط تختصر خارطة التعلم من الألف إلى الياء.

**1**. **المتطلبات المسبقة** , تحتاج إلى:

–  أساسيات الجبر الخطي وحساب التفاضل والتكامل

– لغة البرمجة بايثون: إذا كنت لا تعرف من أين تبدأ في تعلمها، اطلع على قوائم التشغيل Python [fundamentals](https://www.youtube.com/playlist?list=PL-osiE80TeTskrapNbzXhwoFUiLCjGgY7) و  [OOPs concepts](https://www.youtube.com/playlist?list=PL-osiE80TeTsqhIuOqKhwlXsIBIdSeYtc)

**2**. **Deeplearning.ai’s** [**Learning Learning Specialization**](https://www.coursera.org/specializations/deep-learning) ( تجاهل Tensorflow 1.x التي عفا عليها الزمن )

–  سيعطيك هذا جميع الأسس النظرية التي تحتاجها في DL و ML بشكل عام. لا سيما ” [Structuring ML Projects](https://www.coursera.org/learn/machine-learning-projects?specialization=deep-learning) “.

**3**. [**قائمة التشغيل Pytorch Deeplizard**](https://www.youtube.com/playlist?list=PLZbbT5o_s2xrfNyHZsM6ufI0iZENK9xgG)** **

–  هناك منشورات المدونة المصاحبة لها [هنا](https://deeplizard.com/learn/video/v5cngxo4mIg) . تحتوي على برامج تعليمية رائعة حول كيفية إنشاء شبكة ، وكيفية عمل الكود فعليًا ، إلخ.

–  بطيئة بعض الشيء ، ولكن بالتأكيد مفيدة إذا كان لديك الصبر.

**4. قائمة التشغيل** **Fastai’s** [**“Deep learning for coders” playlist**](https://www.youtube.com/playlist?list=PLfYUBJiXbdtSIJb-Qd3pw0cqCbkGeS0xn)

– تحتوي على بعض الرسوم التوضيحية (مثل تلك المتعلقة [بخوارزميات التحسين](https://youtu.be/CJKnDu2dxOE?t=6232) ) فريدة من نوعها وسهلة الفهم. 

**5**. **إذا كنت تريد تعلم معالجة اللغة الطبيعية** 

[jalammar.github.io](https://jalammar.github.io/) (جهاد العمار)

– أحد المعلمين القلائل الذين يقومون بتوصيل المعلومات بطريقة سهلة للغاية وجعلها مبسطة سهلة الفهم. (انظر إلى [هذا](https://jalammar.github.io/images/t/self-attention-matrix-calculation-2.png) وإلى [هذا](https://jalammar.github.io/images/t/self-attention-output.png) أثناء شرحه [transformer networks](https://jalammar.github.io/illustrated-transformer/) )

[قائمة تشغيل Rachel Thomas عن NLP](https://www.youtube.com/playlist?list=PLtmWHNX-gukKocXQOkQjuVxglSDYWsSh9)

– ابدأ [بالفيديو 8](https://www.youtube.com/watch?v=PNNHaQUQqW8&list=PLtmWHNX-gukKocXQOkQjuVxglSDYWsSh9&index=9&t=0s) ، حيث يبدأ الحديث عن نماذج اللغة والبناء من هناك. 

**ونصيحة أخيرة: **

إذا كنت غير صبور وتريد فقط أن ترى كيفية القيام بمعالجة الصور ، فتوقف عند النقطة 4 أعلاه من القائمة . فمن الفيديو الأول تستشعر بكثير من الرضا مما تعلمت.

وإذا لم تكن مبرمج ولا تخطط لتكون كذلك ، [فقم بمراجعة قائمة تشغيل Tech with Tim’s playlist on Tensorflow2.0](https://www.youtube.com/playlist?list=PLzMcBGfZo4-lak7tiFDec5_ZMItiIIfmj).  تغطي Keras وهي وحدة عالية المستوى high level module في Tensorflow 2.0  والتي تسهل كتابة الكود حتى وإن لم تكن مبرمج.

ترجم بتصرف من مصدرين :

1.  Perez, Carlos. “The Deep Learning Roadmap”. *Medium*, 2017, [https://medium.com/intuitionmachine/the-deep-learning-roadmap-f0b4cac7009a](https://medium.com/intuitionmachine/the-deep-learning-roadmap-f0b4cac7009a)
2.  “Deep Learning Roadmap For Beginners”. *Mc.Ai*, 2019, [https://mc.ai/deep-learning-roadmap-for-beginners/](https://mc.ai/deep-learning-roadmap-for-beginners/).
