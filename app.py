import streamlit as st
import os
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

# Load the model and processor
processor = BlipProcessor.from_pretrained("model")
model = BlipForConditionalGeneration.from_pretrained("model")

# Streamlit app title and description
st.title("Radiology Report Generator")
st.write("Upload chest X-ray images and get radiology reports.")
st.markdown(
    """
    Upload your own images or choose from the examples below to generate radiology reports.
    More about this model can be found in my [medium blog](https://medium.com/@sureshnithin1729/interpret-cxrs-using-biobert-clip-456d0ce8cda2).
    """
)

# Upload images
uploaded_files = st.file_uploader("Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Example images folder path
examples_folder = "examples"

# Process each uploaded image
if uploaded_files:
    st.write("**Generated Reports:**")
    
    # Create a row to display images
    images_row = st.empty()

    # Process each uploaded file
    for uploaded_file in uploaded_files:
        # Open image
        image = Image.open(uploaded_file)
        
        # Process the inputs
        inputs = processor(
            images=image, 
            text='',  # Modify with your indication text
            return_tensors="pt"
        )

        # Generate the report
        output = model.generate(**inputs, max_length=512)
        report = processor.decode(output[0], skip_special_tokens=True)
        
        # Display the image and report
        st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
        st.write(f"**Generated Report for {uploaded_file.name}:**")
        # st.write(report)
        try :
            indication=report.split('findings')[0]

            fidx=report.find('findings')
            impdx=report.find('impression')
            findings=report[fidx:impdx]
            impressions=report[impdx:]
            
            st.write(f"{indication}\n")
            st.write(f"{findings}\n")
            st.write(f"{impressions}\n")
        except:
            st.write(report)
        # Append image to the images row with a small gap
        images_row.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
        images_row.markdown("---")  # Adds a small gap between images

# Display example images from folder
st.header("Sample Images ")
st.write("Drag & Drop them in the box above or upload your own files")
example_images = os.listdir(examples_folder)
example_images = os.listdir(examples_folder)
for example_image in example_images:
    # Load example image
    image_path = os.path.join(examples_folder, example_image)
    image = Image.open(image_path)
    
    # Display the example image
    st.image(image, caption=f"Example Image: {example_image}", use_column_width=True)
