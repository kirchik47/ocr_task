import streamlit as st
from main_got import extract_text
import re


def highlight_keywords(text: str, keywords: str) -> str:
    # Split text into sentences
    sentences = text.split('. ')
    highlighted_text = ""
    
    for sentence in sentences:
        highlighted_sentence = sentence
        flag = False
        for keyword in keywords.split():
            if keyword.lower() in sentence.lower():
                # Highlight keyword in yellow and set flag to True to highlight whole sentence after cycle
                highlighted_sentence = highlighted_sentence.replace(keyword, f'<mark style="background-color: yellow">{keyword}</mark>')
                flag = True
        if flag:
            highlighted_text += f'<span style="color: red">{highlighted_sentence}.</span> '
        else:
            highlighted_text += sentence + '. '
    return highlighted_text


# Streamlit UI
st.title("OCR and Document Search Web App")

# Image upload
uploaded_image = st.file_uploader("Upload an image for OCR", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    with st.spinner("Processing..."):
        # Extract text from the uploaded image
        extracted_text = extract_text(uploaded_image)
        st.subheader("Extracted Text")
        st.write(extracted_text)

        # Search functionality
        search_query = st.text_input("Enter a keyword to search within the text")
        if search_query:
            st.markdown(highlight_keywords(extracted_text, search_query), unsafe_allow_html=True)
                