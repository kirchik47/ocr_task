from transformers import AutoModel, AutoTokenizer
import torch
from replacer import replace_escape_sequences


def extract_text(image_path):
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

    image_file = image_path
    # Extract text
    res = model.chat(tokenizer, image_file, ocr_type='ocr')
    return res
