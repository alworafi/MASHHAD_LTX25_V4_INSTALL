# LTX-2.5 DIRECT PARALLEL FAST V2

نسخة مستقلة لتشغيل محرك LTX-2.5 الرسمي مباشرة من Python على RunPod، بلا ComfyUI وبلا Docker Image مخصصة.

## الثوابت

- الصورة الأساسية: `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
- الجذر الدائم: `/workspace/LTX25-DIRECT`
- مجلد الأوزان الوحيد: `/workspace/LTX25-DIRECT/models`
- عدد ملفات الأوزان: خمسة ملفات BF16 فقط.
- لا يحتفظ المثبت بنسخة Hugging Face Cache بعد نجاح التنزيل.
- INT8 ConvRot غير مفعّل ولا يُدّعى دعمه قبل اختبار GPU فعلي مستقل.

## أول تثبيت

يشغّل المثبت مسارين بالتوازي:

1. تنزيل الملفات الخمسة بالتوازي، بحد أقصى خمسة اتصالات ملفات.
2. تثبيت نسخة Git ثابتة من LTX الرسمي وبناء البيئة الدائمة.

يعاد استخدام Torch 2.9.1 وCUDA 12.8 من صورة RunPod ولا ينزلهما المثبت مرة ثانية. يفشل التثبيت بوضوح إذا لم تجتز الصورة أو البطاقة فحص التوافق.

## Runtime readiness truth

The health response deliberately separates `service_ready`, `pipeline_ready`,
`model_ready`, and `model_loaded_to_gpu`. Constructing the official lazy
`DistilledPipeline` never claims that all weights are already resident in VRAM.
`model_ready` becomes true only after a complete real generation succeeds.

The official LTX component lifecycle loads stages on demand and disposes their
parameters back to `meta`; this build therefore keeps the pipeline object and a
warm CUDA allocator, and enables the official in-process RAM weight registry
automatically only when system RAM is at least 96 GB. This accelerates later
requests without falsely claiming full-GPU residency or risking ordinary Pods.

## Fast Resume

عند توصيل نفس Network Volume إلى Pod جديد بالصورة نفسها، يتحقق `install.sh` من البيئة والبطاقة ثم يشغل `start.sh` مباشرة. لا يعيد تثبيت Python أو Torch أو مكتبات LTX، ولا يعيد تنزيل النماذج.

تُحمّل خدمة Direct كائن `DistilledPipeline` مرة واحدة لكل تشغيل Pod وتحتفظ به بين التوليدات. يعرض فحص الصحة حالتين منفصلتين:

- `service_ready`: الـAPI يعمل.
- `model_ready`: الأوزان حُمّلت وأصبح المحرك جاهزًا للتوليد.

على بطاقات الذاكرة الأصغر يختار الوضع التلقائي CPU offload للحفاظ على BF16. على بطاقة ذات هامش VRAM كافٍ (44 GB أو أكثر حاليًا) يستخدم وضع `none` الأسرع.

## أمر القالب

القيم المرجعية موجودة في `runpod-template.json`. يحتاج القالب إلى Network Volume مركب على `/workspace` وإلى `WORKER_SHARED_SECRET` الذي يرسله موقع Mashhad. يمكن إضافة `HF_TOKEN` عندما يتطلب حساب Hugging Face ذلك.

لا ينشئ هذا المشروع Pod أو Network Volume بنفسه.
