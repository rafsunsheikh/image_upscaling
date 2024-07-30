import cv2
import numpy as np
from cv2 import dnn_superres
import streamlit as st

# Create the SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Load the desired model
model_path_1 = "EDSR_x4.pb"
model_path_2 = "LapSRN_x8.pb"
model_selection = st.selectbox("Select Model", ["EDSR x4", "LapSRN x8"])
if model_selection == "EDSR x4":
    sr.readModel(model_path_1)
    sr.setModel("edsr", 4)
elif model_selection == "LapSRN x8":
    sr.readModel(model_path_2)
    sr.setModel("lapsrn", 8)

# Function to upscale the image
def upscale_image(image):
    result = sr.upsample(image)
    return result

# Streamlit app
def main():
    st.title("Image Upscaling with " + model_selection)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        # image = cv2.imread(uploaded_file.read())
        
        # Upscale the image
        upscaled_image = upscale_image(image)
        
        # Display the original and upscaled images
        st.image([image, upscaled_image], caption=["Original Image", "Upscaled Image"], width=300)
        
        # Save the upscaled image
        st.markdown("### Download Upscaled Image")
        st.download_button("Download", upscaled_image, file_name="upscaled_image.png")
    
if __name__ == "__main__":
    main()

