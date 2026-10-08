---
title: "توليد صور وجوه مقاربة للحقيقية باستخدام الذكاء الاصطناعي"
date: 2019-11-09
permalink: "/ar/posts/2019/11/ai-generated-faces/"
excerpt: "جولة في تطور شبكات الخصومة التوليدية عبر ثلاث أوراق علمية: GAN وProGAN وStyleGAN، وكيف أصبح الذكاء الاصطناعي قادراً على توليد وجوه لأشخاص لا وجود لهم."
canonical_url: "https://nmthgiat.com/%d8%aa%d9%88%d9%84%d9%8a%d8%af-%d8%b5%d9%88%d8%b1-%d9%88%d8%ac%d9%88%d9%87-%d9%85%d9%82%d8%a7%d8%b1%d8%a8%d8%a9-%d9%84%d9%84%d8%ad%d9%82%d9%8a%d9%82%d9%8a%d8%a9-%d8%a8%d8%a7%d8%b3%d8%aa%d8%ae%d8%af/"
tags:
  - GANs
  - StyleGAN
  - Deep Learning
  - Computer Vision
---

> نُشر هذا المقال أولاً في [نمذجيات](https://nmthgiat.com/%d8%aa%d9%88%d9%84%d9%8a%d8%af-%d8%b5%d9%88%d8%b1-%d9%88%d8%ac%d9%88%d9%87-%d9%85%d9%82%d8%a7%d8%b1%d8%a8%d8%a9-%d9%84%d9%84%d8%ad%d9%82%d9%8a%d9%82%d9%8a%d8%a9-%d8%a8%d8%a7%d8%b3%d8%aa%d8%ae%d8%af/) بتاريخ 9 نوفمبر 2019.

فاجأنا الذكاء الاصطناعي قبل فترة بمقدرة جديدة له على توليد صور أشخاص لا وجود لهم، وأحدث ذلك ضجة ما بين متخوف ومعجب ومتابع بشغف ومتابع بقلق. وتسارعت عجلة التطور لنرى مقاطع فيديو معدلة وتطبيقات تغير صور مستخدميها ليبدون أكثر أو أقل تقدماً بالعمر.

هنا عزيزي القارئ سنأخذك بجولة حول هذه التقنية المتسارعة بالتطور ونحاول سوياً أن نكتشف خباياها وكيف تعمل! وبالتأكيد لا نستطيع أن نغطي كل التطور السريع لهذة التقنية والأبحاث التي نشرت حولها، لذا أخترت **ثلاثة أوراق علمية** مرتبطات ببعضهن وأعتقد إنهن يمثلن أهم مراحل تطور هذه التقنية.

1.  الورقة الأولى تمثل مرحلة البداية والنشأة **GAN** نشرت في يونيو ٢٠١٤. للإطلاع على الورقة اضغط [هنا](https://arxiv.org/abs/1406.2661) \[1\].
2.  ثم تلتها ورقة تمثل مرحلة النمو والتحسين **ProGAN** نشرت في أكتوبر ٢٠١٧ وقدمت تحسينات جيدة لدقة الصور. للإطلاع على الورقة اضغط [هنا](https://arxiv.org/abs/1710.10196) \[2\].
3.  ثم سنختم بورقة تمثل مرحلة النضج النسبي **StyleGAN** نشرت في ديسمبر ٢٠١٨ والتي فتحت آفاق جديدة للتحكم بسمات الصور. للإطلاع على الورقة اضغط [هنا](https://arxiv.org/abs/1812.04948) \[3\].

### مرحلة البداية والنشأة

#### Generative Adversarial Networks – GANs

#### شبكات الخصومة التوليدية GANs : ماهي وكيف نشأت؟

هي تقنية نستطيع وصفها بالمذهلة فهي قادرة على إنشاء صور ومقاطع فيديو وصوت كأنها حقيقية، وهي واحدة من أهم الابتكارات في مجال التعلم العميق التي حققت نجاحًا كبيرًا منذ تقديمها في عام 2014 من قبل باحث يدعى إيان جودفيلو Ian Goodfellow وزملاؤه في جامعة مونتريال. وتعتبر نوع من أنواع التعلم غير الموجّه (unsupervised learning).

تتكون من شبكتين عصبيتين منفصلتين تتنافسان باستمرار ضد بعضهما البعض (مثل لعبة القط والفأر) وهما **شبكة التوليد Generator** والتي تحاول أن تولد أمثلة تبدو حقيقية، فتبذل قصارى جهدها لخداع الأخرى. و **شبكة التمييز Discriminator** التي بدورها تحاول التمييز بين الحقيقي والوهمي، فتبذل قصارى جهدها حتى لا تنخدع. يتم إدخال متجه من الأرقام العشوائية للمولد والمخرجات منه تكون بيانات مولدة غير حقيقية، أما المميز يتم إدخال له بيانات حقيقية والبيانات الخارجة من المولد ويخرج المميز إحتمال كون البيانات المدخلة إليه حقيقية أم مزيفة.

![](/images/posts/ai-generated-faces/01.png)

المصدر \[4\]

##### التحديات والقيود

على الرغم من أن شبكات GAN رائعة نظرياً ومثيرة للإهتمام، ولكن في الممارسة العملية صعبة للغاية في كثير من الأحيان، حيث يفوز المميِز عندما يتوقف المولد عن قدرته على التحسن وإخراج صور شبه حقيقية أو كبيرة بجودة عالية. لذا كانت هناك العديد من المحاولات لجعل المولد يتفوق بإنشاء صور كبيرة عالية الجودة، ومن هذه المحاولات ProGAN.

### مرحلة النمو والتحسين

#### Progressive Growing of GANs for Improved Quality, Stability, and Variation – ProGAN

#### النمو التقدمي لشبكات GAN لتحسين الجودة والاستقرار والتنوع – ProGAN

تمكّن شبكات GAN ذات النمو التقدمي (ProGAN) من الحصول على حجم أكبر وأكثر استقرارًا، حيث نجح فريق NVIDIA بتوليد صور 1024 × 1024 أكثر دقة. الفكرة الأساسية هي تدريب المولد والمميز مع صورة منخفضة الدقة للغاية في البداية (4 × 4)، ومن ثم يضاف طبقة دقة أعلى في كل مرة بشكل تدريجي. الطبقات الجديدة المضافة تحتوي على تفاصيل أكثر دقة، ومع تقدم التدريب يتمكن هذا النموذج من إنتاج صور بجودة عالية، ولعل الصور أدناه تغني عن شرح آلية التحسين في نموذج ProGAN.

![](/images/posts/ai-generated-faces/02.png)

المصدر \[4\]

<video src="/images/posts/ai-generated-faces/03.mp4" poster="/images/posts/ai-generated-faces/03-poster.jpg" autoplay loop muted playsinline aria-label=""></video>

المصدر \[4\]

##### التحديات والقيود

رغم جودة الصور فإن القدرة على التحكم في ميزات وسمات معينة للصورة التي تم إنشاؤها محدودة للغاية، وبالتالي فإن محاولة تعديل المدخلات ولو حتى قليلاً تؤثر عادةً على ميزات متعددة في نفس الوقت. فعلى سبيل المثال، نريد أن نتحكم بتغيير تصفيفة الشعر لشخص في صورة معينة دون أن نغير شكله ولم يكن ذلك ممكنا، ومن هنا  ظهرت الحاجة لمعرفة كيف نتحكم بسمات الصور.

### مرحلة النضج النسبي

#### A Style-Based Generator Architecture for Generative Adversarial Networks – StyleGAN

#### بنية مولد تعتمد على النمط لشبكات الخصومة التوليدية

لحل التحديات السابقة اقترح الباحثين من NVIDIA بنية جديدة **لشبكة التوليد** مع بقاء شبكة التمييز على ما كانت عليه في نموذج ProGAN السابق وذلك من أجل توفير طريقة جديدة للتحكم في عملية إنشاء الصور. واعتمد الباحثون على ما يسمى نقل النمط style transfer. يقوم المولد في StyleGAN بإجراء تعديلات صغيرة على نمط الصورة في كل طبقة من أجل معالجة ميزات الصورة لتلك الطبقة. وتم فصل السمات عالية المستوى مثل هوية الشخص عن الصفات ذات المستوى المنخفض مثل تصفيفة الشعر داخل الصورة ما يسمح بتغيير بعض السمات دون أن تتأثر بقية السمات الأساسية الأخرى.

![](/images/posts/ai-generated-faces/04.png)

المصدر \[4\]

شبكة التوليد أهم ما يميزها من اضافات **Mapping Network** و **Adaptive Normalization – AdaIN** والتي متوقع أن تكون محور للبحث والتطوير المستقبلي. كذلك أصبح يضاف للمولد بعض الضوضاء المتزايدة **Noise** وتم استبدال الإدخال التقليدي للشبكة التوليفية بإدخالات ثابتة.

##### قسمت الورقة السمات أو الميزات إلى ثلاثة أنواع

1. **الخشنة-العامة Coarse** – دقتها  تصل إلى 8×8

-تؤثر على تشكل ستايل الشعر العام و شكل الوجه وبعض السمات الأساسية.

2. ا**لمتوسطة Middle** – دقتها ما بين 16×16 إلى 32×32

-تؤثر على ملامح الوجه الدقيقة مثل شكل عيون واسعة أو ضيقة الخ.

3. **الدقيقة Fine** – دقتها من 64×64 إلى 1024×1024

-تؤثر على نظام الألوان مثل لون العين والشعر والبشرة والميزات الدقيقة الأخرى.

<video src="/images/posts/ai-generated-faces/05.mp4" poster="/images/posts/ai-generated-faces/05-poster.jpg" autoplay loop muted playsinline aria-label=""></video>

المصدر \[5\]

واستخدم أسلوب الخلط Style mixing، وللتوضيح لنفرض أن لدينا المصدر (أ) الذي يمثل الجنس ذكر أو أنثى، العمر، طول الشعر، يرتدي نظارة ..الخ، والمصدر (ب) الذي يمثل بقية الصفات الأخرى، وأردنا أن نولد صور بإستخدام المصدرين سيبدو الناتج كما توضح الصورة أدناه:

<video src="/images/posts/ai-generated-faces/06.mp4" poster="/images/posts/ai-generated-faces/06-poster.jpg" autoplay loop muted playsinline aria-label=""></video>

المصدر \[5\]

وهذه الصورة تشرح مزيدا من التفاصيل عن اسلوب الخلط والسمات الثلاثة التي تم ذكرها أعلاه:

![](/images/posts/ai-generated-faces/07.webp)

المصدر \[3\]

شاهد الفيديو الرسمي لهذه الورقة والذي يستعرض الحلول التي قدمت للتحكم بالصور المولدة لمزيد من التوضيح:

[▶ مشاهدة الفيديو على YouTube](https://www.youtube.com/watch?v=kSLJriaOumA)

### تدريب نموذجك الخاص

تصفح مستودع StyleGAN [مفتوح المصدر](https://github.com/NVlabs/stylegan) يمكنك المساهمة فيه أو تدريب نموذجك الخاص. ولعلك تسأل عزيزي القارئ خصوصاً إذا ما كنت تفكر بتجهيز مجموعة بيانات خاصة بك حول مجموعة البيانات التي درب الباحثون نموذجهم عليها. يمكنك إلقاء نظرة على مجموعة بيانات [ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) والتي تحتوي على 70000 صورة عالية الجودة من وجوه أشخاص بدقة 1024 من موقع Flickr.

وقبل البدء بتدريب نموذجك الخاص تأكد من توفر [متطلبات النظام المحددة](https://github.com/NVlabs/stylegan#system-requirements) مثل GPU، وقد تم تقدير الوقت المتوقع للتدريب كما بالجدول التالي:

![](/images/posts/ai-generated-faces/08.png)

المصدر \[3\]

كما هو موضح تدريب نموذجك الخاص مكلف لذا دعونا نستخدم **النموذج المدرب مسبقا** لتوليد صور شبه حقيقية في جزء التطبيق العملي

#### جرب إنشاء صورة بنفسك

افتح [Google Colabs](https://colab.research.google.com/) ومن ثم اختار استخدام GPU

انسخ المستودع التالي

    !git clone https://github.com/NVlabs/stylegan.git

خزن مسار المجلد

``` Python
import sys
sys.path.insert(0, "/content/stylegan")
```

``` Python
## Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
## This work is licensed under the Creative Commons Attribution-NonCommercial
## 4.0 International License. To view a copy of this license, visit
## http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
## Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

def main():
    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network.
    url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
    with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
        _G, _D, Gs = pickle.load(f)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    # Pick latent vector.
    rnd = np.random.RandomState()
    

    latents = rnd.randn(1, Gs.input_shape[1])

    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)

    # Save image.
    os.makedirs(config.result_dir, exist_ok=True)
    png_filename = os.path.join(config.result_dir, f'/content/example.png')
    PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

if __name__ == "__main__":
    main()
```

قم باستعراض الصورة المتولدة

``` Python
from IPython.display import Image
Image('/content/example.png')
```

**كامل الكود** موجود على هذا [**الرابط**](https://github.com/Ruqyai/publish/blob/master/StyleGAN.ipynb) ويمكنك الاستمتاع بتصفح هذا الموقع [This Person Does Not Exist](https://thispersondoesnotexist.com/) لمعاينة بعض الصور التي تم توليدها ب StyleGAN وكلما حدثت الصفحة ستتولد صورة جديدة.

### خاتمة

نختم بالجدل المثار حول هذه التقنيات وإختلاف وجهات النظر ما بين نظرة سلبية وأخرى إيجابية، ونترك لك الحكم .. فإلى أي كفتي وجهات النظر ستميل؟!

##### حجج أصحاب النظرة السلبية

يرى أصحاب النظرة السلبية بأن تطور مثل هذه التقنيات يجعلنا نفقد الثقة بحواسنا فيما نراه أو نسمعه ويزرع الشك الخوف وانعدام الثقة، فمن الصعب التمييز بين الحقيقة والكذب والتدليس. ويرون أن تطور هذه التقنيات باب ومدخل لتسهيل نشر الإشاعات والأكاذيب، وبل وقد يمتد ضررها لتسهيل حدوث الجرائم.

##### حجج أصحاب النظرة الإيجابية

يرون أن هناك مبالغة في المخاوف، توجد من قبل أدوات مثل برامج تحرير الفيديو والصور، ومن يريد احتراف الخداع والتدليس يستطيع استخدامه. وأنها تعتبر مثل أي أداة سلاح ذو حدين لو استخدمت بشكل صحيح يمكن أن تخدم تطور مجال التصميم والجرافكس، صناعة السينما، الإعلام، ألعاب الفيديو، الإعلانات التجارية وغيرها من المجالات.

### المصادر

مصدر صورة المقال الرئيسية \[3\].

1.  Goodfellow, Ian, et al. “Generative adversarial nets.” Advances in neural information processing systems. 2014.‏

&nbsp;

2.   Karras, Tero, et al. “Progressive growing of gans for improved quality, stability, and variation.” arXiv preprint arXiv:1710.10196 (2017).‏
3.   Karras, Tero, Samuli Laine, and Timo Aila. “A style-based generator architecture for generative adversarial networks.” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.‏
4.   Horev R. Style-based GANs – Generating and Tuning Realistic Artificial Faces \| Lyrn.AI. Lyrn.AI. https://www.lyrn.ai/2018/12/26/a-style-based-generator-architecture-for-generative-adversarial-networks/. Published 2018. Accessed October 29, 2019.
5.   Karras, Tero. “A Style-Based Generator Architecture For Generative Adversarial Networks”. Youtube, 2019, https://www.youtube.com/watch?v=kSLJriaOumA&feature=youtu.be. Accessed 3 Nov 2019.
