# Image Upscaling Application

This repository contains an Image Upscaling Application that utilizes deep learning models to enhance the resolution of images. The application is built using OpenCV and Streamlit, providing a simple and interactive web interface for users to upload and upscale their images.

## Features

- **Two Upscaling Models**: Choose between EDSR (x4) and LapSRN (x8) models.
- **Interactive Web Interface**: Easy-to-use interface for uploading and viewing images.
- **Image Comparison**: Display original and upscaled images side by side.
- **Download Functionality**: Option to download the upscaled image.

## Requirements

- Python 3.7+
- OpenCV
- Streamlit
- NumPy

## Installation

1. Clone the repository:
    ```sh
    git clone [https://github.com/your-username/image-upscaling-app.git](https://github.com/rafsunsheikh/image_upscaling.git)
    cd image_upscaling
    ```

2. Install the required packages:
    ```sh
    pip install opencv-contrib-python streamlit numpy
    ```

3. Download the models and place them in the repository directory:
    - [EDSR x4 model](https://example.com/EDSR_x4.pb)
    - [LapSRN x8 model](https://example.com/LapSRN_x8.pb)

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload an image and select the desired upscaling model. The application will display the original and upscaled images side by side. You can also download the upscaled image.

## File Structure

- `app.py`: Main application script.
- `EDSR_x4.pb`: EDSR x4 upscaling model.
- `LapSRN_x8.pb`: LapSRN x8 upscaling model.

## Code Overview

### `app.py`

The main application script includes the following key components:

- **Model Loading**: Loads the selected model based on user input.
- **Upscaling Function**: Uses the loaded model to upscale the uploaded image.
- **Streamlit Interface**: Handles file uploads, displays images, and provides a download button for the upscaled image.

```python
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
        
        # Upscale the image
        upscaled_image = upscale_image(image)
        
        # Display the original and upscaled images
        st.image([image, upscaled_image], caption=["Original Image", "Upscaled Image"], width=300)
        
        # Save the upscaled image
        st.markdown("### Download Upscaled Image")
        st.download_button("Download", upscaled_image, file_name="upscaled_image.png")
    
if __name__ == "__main__":
    main()
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
