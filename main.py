from transformers import AutoModel, AutoTokenizer
import torch
from byaldi.RAGModel import RAGMultiModalModel
from byaldi.colpali import ColPaliModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import numpy as np


colpali_model = ColPaliModel.from_pretrained('vidore/colpali')
print(colpali_model.doc_id_to_metadata)
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": Image.open('template.jpg'),
            },
            {"type": "text", "text": 'Return full text of the document as a plain text'},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
img = Image.open('docs/hindi_template.jpg')
inputs = processor(
    text=text,
    images=img,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cpu")
generated_ids = model.generate(**inputs, max_new_tokens=5000)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
