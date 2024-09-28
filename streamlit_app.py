import streamlit as st
from main_got import extract_text
import re

# Streamlit UI
st.title("OCR and Document Search Web App")

# Image upload
uploaded_image = st.file_uploader("Upload an image for OCR", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    with st.spinner("Processing image..."):
        # Extract text from the uploaded image
        extracted_text = extract_text(uploaded_image)
        st.subheader("Extracted Text")
        st.write(extracted_text)

        # Search functionality
        search_query = st.text_input("Enter a keyword to search within the text")
        if search_query:
            results = [match.start() for match in re.finditer(search_query, extracted_text)]
            if results:
                st.subheader("Search Results")
                for result in results:
                    st.write(f"Keyword found at index: {result}")
            else:
                st.write("No results found.")
                