# ocr_task
OCR assignment for PARIMAL IIT Roorkee Internship.

Two models for OCR were considered: GOT 2.0 and Colpali implementation of Byaldi library + Qwen2-VL. After research GOT was chosen because it has specification of extracting text from image directly without using LLM for explaining the content of the file. Besides that, GOT has direct instructions for training and fine-tuning model with data samples. Since GOT does not generate hindi symbols at all, I've needed to fine-tune the model on hindi dataset. Tokenizer already contained tokens for hindi symbols, so adding tokens was not necessary. 
I've chosen to use Kaggle Notebook for this since it provides powerful GPU for limited use.

During deployment to streamlit on huggingface space encountered a problem with '\left' strings which were problematic escape sequences due to '\'. Cloned model repository and directly changed the code. Also handled cpu compatibility issues and other minor bugs.

WARNING: 
Application works pretty slow on CPU, so if you upload for example resume, it will be proceeded for approximately 10 minutes. Didn't test it on CUDA because don't have it.

For training used:
!git clone https://github.com/modelscope/ms-swift.git
%cd ms-swift
!pip install -e .[llm]

!swift sft \
--model_type got-ocr2 \
--model_id_or_path stepfun-ai/GOT-OCR2_0 \
--sft_type lora \
--dataset /kaggle/input/json-images/dataset.json

Dataset used for fine-tuning:
https://www.kaggle.com/datasets/prathmeshzade/hindi-ocr-synthetic-line-image-text-pair
Application link:
https://huggingface.co/spaces/kirchik47/ocr_task
