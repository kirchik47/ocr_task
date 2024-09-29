# ocr_task
OCR assignment for PARIMAL IIT Roorkee Internship.

Two models for OCR were considered: GOT 2.0 and Colpali implementation of Byaldi library + Qwen2-VL. After research GOT was chosen because it has specification of extracting text from image directly without using LLM for explaining the content of the file. Besides that, GOT has direct instructions for training and fine-tuning model with data samples. Since GOT does not generate hindi symbols at all, I've needed to fine-tune the model on hindi dataset. Tokenizer already contained tokens for hindi symbols, so adding tokens was not necessary. 
However, GOT is only compatible with CUDA, so on my device it won't be possible to fine-tune it. I've chosen to use Google Colab for this since it provides GPU for limited use.

During deployment on streamlit sharing encountered a problem with '\left' strings which were problematic escape sequences due to '\'. Used additional script replacer.py to replace all these string to '\\left'.

