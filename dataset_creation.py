import pandas as pd
import json
import os


dataset = pd.read_csv('data_80k/data.csv')
labels = dataset['image_file']
text = dataset['text']
json_data = []
images_path = '/kaggle/input/hindi-english-images/data_80k/output_images/'
for i in range(len(labels)):
    json_data.append(
        {
            "query": "<image>",
            "response": text[i],
            "images": [os.path.join(images_path, labels[i])],
        }
    )
with open('dataset.json', 'w') as f:
    json.dump(json_data, f)
    