import os

def replace_escape_sequences(seq):
    # Path to the file that needs modification
    file_path = "/home/appuser/.cache/huggingface/modules/transformers_modules/ucaslcl/GOT-OCR2_0/cf6b7386bc89a54f09785612ba74cb12de6fa17c/modeling_GOT.py"

    # Read the file
    with open(file_path, 'r') as file:
        file_data = file.read()

    # Define replacements for problematic escape sequences
    old = seq
    new = '\\' + old

    # Perform the replacements
    file_data = file_data.replace(old, new)

    # Write the modified content back to the file
    with open(file_path, 'w') as file:
        file.write(file_data)

    print(f"Replacements done successfully in {file_path}")
    