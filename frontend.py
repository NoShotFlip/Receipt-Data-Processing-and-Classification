import json
import pytesseract
import requests
import streamlit as st
from PIL import Image
import re


st.title("Image Uploader and Text Extractor")

# Allow users to upload multiple image files
uploaded_file = st.file_uploader("Choose an image...", type=[
                                 "jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_file is not None:
    for uploaded_file in uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        extracted_text = pytesseract.image_to_string(image)

        st.write("Extracted Text:")
        st.write(extracted_text)

        image_binary = uploaded_file.getvalue()

        # Model api URL (You have to update this part)
        api_url = "https://7hdkpvjtj3.execute-api.us-east-1.amazonaws.com/v1/receipts"

        headers = {
            'Content-Type': 'image/jpeg'
        }

        # Send the binary image data in the POST request
        response = requests.post(api_url, headers=headers, data=image_binary)

        if response.status_code == 200:

            transformed_output = response.json()
            st.write("Transformed Output:")
            st.json(transformed_output)

            # This part will be the nice formatting part using the json (transformed_output)
        else:
            st.write(
                "Error: Could not get a response from the model. Please try again later.")

        # Initialize a dictionary to store the formatted data
        formatted_data = {

            "Receipt_text": extracted_text,
            "Item_list": []

        }

        line = extracted_text.splitlines()
        for line in line:

            # Use a regular expression to match lines that contain prices
            if re.search(r'\$\d+\.\d{2}', line) or re.search(r'\d+\.\d{2}', line):

                formatted_data["Item_list"].append(line.strip())

        st.write("formatted_data: ")
        st.json(formatted_data)


else:
    st.write("Please upload an image file.")