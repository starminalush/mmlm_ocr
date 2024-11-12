# Цель проекта:
Реализовать решение по распознаванию значимой информации на документах(товарные накладные, договоры купли-продажи, т.д.)
Примеры данных:
1. [Накладная на товар](images/blank-torg12-03.png)
2. [Счет на оплату](images/4-blankscheta-new.png)
3. [Паспорт](images/img-48999-15003795646755.jpg)

#  Технические ограничения на данном этапе:
Нужна одна мультимодальная нейросетевая модель, которая могла бы по заданному текстовому запросу извлекать данные. Решения в виде связки OCR + NER не подходят

# Ресерч существующих решений

Было найдено несколько бенчмарков, подходящих под данную задачу:
- [OCRBench Leaderboard](https://huggingface.co/spaces/echo840/ocrbench-leaderboard)
это комплексный оценочный бенчмарк, разработанный для оценки возможностей оптического распознавания символов (OCR) у крупных мультимодальных моделей. Он включает пять компонентов: распознавание текста, VQA с текстом в сценах, VQA с документами, извлечение ключевой информации и распознавание рукописных математических выражений. В бенчмарке содержится 1000 пар вопрос-ответ, и все ответы проходят ручную проверку и корректировку для обеспечения более точной оценки.
- [SROIE](https://paperswithcode.com/dataset/sroie) - датасет, представленный на конференции ICDAR2019 Competition on Scanned Receipt OCR and Information Extraction, содержит в себе фотографии чеков из магазина
- [FUNSD](https://guillaumejaume.github.io/FUNSD/) это датасет, который содержит сканы документов
- [CORD](https://github.com/clovaai/cord) - тоже датасет с рецептами, но больше вариативности данных
- [DocVQA](https://www.docvqa.org/) - датасет с документами
- [LLMArena-Vision](https://lmarena.ai/) - на llmarena есть вкладка для моделей, котоорые умеют распознавать образы
- [MTVQA](https://github.com/bytedance/MTVQA) - мультиязычный фреймворк, там есть русский язык

# Первичные модели кандидаты для дальнейшего ресерча:
Рассматриваются либо квантизированные модели, либо модели, которые помещаются в 30гб памяти, потому что пока у меня только столько есть.
Для проверки, сколько VRAM нужно модели, использовался [llmexplorer](https://llm.extractum.io/).
 - Llava-next(aka Llava1.6)
   - [Llava V1.6 Mistral 7B](https://llm.extractum.io/model/liuhaotian%2Fllava-v1.6-mistral-7b,4c7bhrLggBCQDUbcMAfxID)
   - [Llava V1.6 Vicuna 7B](https://llm.extractum.io/model/liuhaotian%2Fllava-v1.6-vicuna-7b,5GHEETz8QYZp4RvAinuxKX)
   - [Llava 1.6 Gptq 8bit](https://llm.extractum.io/model/panoyo9829%2Fllava-1.6-gptq-8bit,22w2Pm5cLo5tGDixGIy6tF)
   - [Candle Llava V1.6 Mistral 7B](https://llm.extractum.io/model/DanielClough%2FCandle_llava-v1.6-mistral-7b,5wz85432jEFN1pMVVsaAOf)
   - [llava-v1.6-34b.Q5_K_M.gguf](https://huggingface.co/cjpais/llava-v1.6-34B-gguf)
 - Donut
   - [donut-base-finetuned-docvqa](naver-clova-ix/donut-base-finetuned-docvqa)
   - [donut-base](naver-clova-ix/donut-base)
 - Molmo
   - [Molmo 7B D Bnb 4bit](https://llm.extractum.io/model/cyan2k%2Fmolmo-7B-D-bnb-4bit,1XyyBGljOhcw89K6FheXKN)
   - [Molmo 7B O Bnb 4bit](https://llm.extractum.io/model/cyan2k%2Fmolmo-7B-O-bnb-4bit,1kU9Getow6BZan0obQ4HMH)
 - MiniCRM-v2.6
   - [MiniCPM-V-2_6](https://huggingface.co/openbmb/MiniCPM-V-2_6)
 - Pixtral
   - [mistral-community/pixtral-12b](https://huggingface.co/mistral-community/pixtral-12b)
 - PaliGemma
   - [paligemma_vqav2](https://huggingface.co/merve/paligemma_vqav2)
 - Llava Saiga
   - [llava-saiga-8b](https://huggingface.co/deepvk/llava-saiga-8b)
 - Fuyu
   - [fuyu-8b](https://huggingface.co/adept/fuyu-8b)
 - Phi-3-Vision
   - [Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)
 - Chameleon
   - [chameleon-7b](https://huggingface.co/facebook/chameleon-7b)
 - H2OVL-Mississippi
   - [h2oai/h2ovl-mississippi-800m](https://huggingface.co/h2oai/h2ovl-mississippi-800m)
   - [h2oai/h2ovl-mississippi-2b](https://huggingface.co/h2oai/h2ovl-mississippi-2b)
 - InternVL
   - [OpenGVLab/InternVL2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B)
 - Qwen2-VL
   - [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
 - Idefics3-8B-Llama3
   - [HuggingFaceM4/Idefics3-8B-Llama3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3)
 - Vikhr
   - [Vikhrmodels/Vikhr-2-VL-2b-Instruct-experimental](https://huggingface.co/Vikhrmodels/Vikhr-2-VL-2b-Instruct-experimental)
 - Monkey
   - [echo840/Monkey](https://huggingface.co/echo840/Monkey)
 - BLIVA
   - [mlpc-lab/BLIVA_Vicuna](https://huggingface.co/mlpc-lab/BLIVA_Vicuna)
 - RuLlava
   - [ruIdefics2-ruLLaVA-merged](https://huggingface.co/GeorgeBredis/ruIdefics2-ruLLaVA-merged)

Принцип проверки первичных кандидатов:
 - найти информацию, могут ли они понимать русский язык
 - протестировать запуск и попробовать извлечь информацию с тестовых картинок, если есть такая возможность

# Статьи
- [VLM в Нейро: как мы создавали мультимодальную нейросеть для поиска по картинкам](https://habr.com/ru/companies/yandex/articles/847706/)
- [ICDAR 2024](https://icdar2024.net/competitions/)
