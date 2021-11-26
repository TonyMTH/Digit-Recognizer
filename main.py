import streamlit as st
from predict import Predict
from PIL import Image

model_pth = 'data/best_model.pt'
st.title("Handwriting Recognizer\n\n")

uploaded_file = st.file_uploader('Upload Image File')
if uploaded_file is not None:
    kernel_size = st.slider('Kernel Size', min_value=1, max_value=50, value=5, step=1)
    min_thresh = st.slider('Minimum Threshold', min_value=0, max_value=255, value=116)
    n = st.slider('Number of Zero Padding Around Image', min_value=1, max_value=100, value=21)

    selected_img = st.radio("What image do you want to use?", ('Inverted', 'Original'))
    if selected_img == 'Original':
        original_image = True
    else:
        original_image = False

    image = Image.open(uploaded_file).convert('L')
    st.image(image, channels="BGR", width=300)

    prob, predicted, img, raw_im = Predict(model_pth, uploaded_file).predict(original_image, kernel_size, min_thresh, n, dim=28)
    # st.image(img, channels="L", width=100)
    st.image(img, width=50, use_column_width='auto')
    st.image(raw_im, use_column_width='auto')

    st.write(f'The image passed is {predicted} with average probability of {prob}%')
else:
    st.write(f'Please upload a file')
