from transformers import AutoModel, AutoTokenizer
import torch


if torch.cuda.is_available():
    device = torch.device('cuda') # If cuda is available, use it, otherwise use CPU
else:
    device = torch.device('cpu')

tokenizer = AutoTokenizer.from_pretrained('ucaslcl/GOT-OCR2_0', 
                                          trust_remote_code=True # Allows custom code to load model from hub
            )
model = AutoModel.from_pretrained('ucaslcl/GOT-OCR2_0', 
                                  trust_remote_code=True, 
                                  low_cpu_mem_usage=True, 
                                  device_map=device.type, 
                                  use_safetensors=True, # This format is faster, more memory efficient 
                                                        # and provides safe deserialization unlike pickle-based one
                                  pad_token_id=tokenizer.eos_token_id # Set the pad token from tokenizer
        )


# input your test image
image_file = 'docs/template.jpg'

res = model.chat(tokenizer, image_file, ocr_type='ocr')

# format texts OCR:
# res = model.chat(tokenizer, image_file, ocr_type='format')

# fine-grained OCR:
# res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_box='')
# res = model.chat(tokenizer, image_file, ocr_type='format', ocr_box='')
# res = model.chat(tokenizer, image_file, ocr_type='ocr', ocr_color='')
# res = model.chat(tokenizer, image_file, ocr_type='format', ocr_color='')

# multi-crop OCR:
# res = model.chat_crop(tokenizer, image_file, ocr_type='ocr')
# res = model.chat_crop(tokenizer, image_file, ocr_type='format')

# render the formatted OCR results:
# res = model.chat(tokenizer, image_file, ocr_type='format', render=True, save_render_file = './demo.html')

print(res)