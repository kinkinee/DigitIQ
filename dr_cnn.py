import tensorflow as tf
import numpy as np
from matplotlib import pyplot
import streamlit as st
from PIL import Image
import cv2
from streamlit_drawable_canvas import st_canvas
import os
import subprocess

st.set_page_config(page_title = "Digit Recognition")
st.title("Digit Recognition")

feedback_count = 0
for root, dirs, files in os.walk("my_digits"):
    feedback_count += len(files)
st.sidebar.header("Dataset")
st.sidebar.write(f"Images: {feedback_count}")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
df = ((x_train, y_train), (x_test, y_test))

x_train = (x_train/255)
x_test = (x_test/255)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(30, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
#model.summary()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("digit_cnn_combined.keras")

import time
status = st.empty()
status.info("Summoning the Prediction God...")
model = load_model()
time.sleep(3)
status.success("The Prediction God is here.")
time.sleep(4)
status.empty()

test_loss, test_accuracy = model.evaluate(x_test, y_test)
st.metric(
    label="Model Accuracy",
    value=f"{test_accuracy * 100:.2f}%"
)


uploaded_file = None
canvas_result = None

mode = st.radio(
    "Input Method",
    ["Upload Image", "Draw Digit"],
    horizontal=True
)

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload a digit image", type=["png","jpg","jpeg"])

if mode == "Draw Digit":
    canvas_result = st_canvas(fill_color="rgba(0,0,0,0)",
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas")

cropped = None
feedback = False
if mode == "Upload Image" and uploaded_file is not None:
    st.write("File uploaded successfully")
    image = Image.open(uploaded_file)
    image = image.convert('L')  # Grayscale
    img_array = np.array(image)
    if np.mean(img_array) < 127:
        img_array = 255 - img_array
    _, thresh = cv2.threshold(img_array, 50, 255, cv2.THRESH_BINARY)
    

    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        display_img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(display_img, (x,y), (x+w,y+h), (255,0,0), 2)
        cropped = thresh[y:y+h,x:x+w]

elif (mode == "Draw Digit"
    and canvas_result is not None
    and canvas_result.image_data is not None
    and np.any(canvas_result.image_data[:, :, :3] != 255)):
        img = canvas_result.image_data

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
        #st.subheader("Canvas Grayscale")
        #st.image(img)
        #img = 255 - img
        #st.subheader("Canvas Inverted")
        #st.image(img)
        _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        #st.subheader("Canvas Threshold")
        #st.image(thresh)
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            cropped = thresh[y:y+h,x:x+w]

if cropped is not None:
    h, w = cropped.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)

    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    square[ y_offset:y_offset+h, x_offset:x_offset+w ] = cropped

    square = cv2.copyMakeBorder(
        square, 4, 4, 4, 4,
        cv2.BORDER_CONSTANT,
        value=0)

    cropped = cv2.resize(square, (28,28))
    #st.write("Shape before CNN:", cropped.shape)
    st.subheader("Image fed to CNN")
    st.image(cropped, width=200)  
  
    cropped = 255 - cropped
    cropped = cropped / 255.0
    cropped = cropped.reshape(1, 28, 28, 1)
    
    prediction = model.predict(cropped,verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    if confidence > 50:
        st.success(f"Confidence: {confidence:.2f}%")
    else:
        st.warning(f"Low confidence: {confidence:.2f}%")
    
    st.metric("Predicted Digit:", predicted_digit)
    with st.expander("Wrong Prediction?"):
        feedback = st.checkbox("Provide Feedback")
    

    if feedback:
        actual_digit = st.selectbox("Select correct digit", list(range(10)))
        if st.button("Submit Feedback"):
            folder = f"my_digits/{actual_digit}"
            os.makedirs(folder, exist_ok=True)
            import uuid
            filename = os.path.join(folder, f"{uuid.uuid4()}.png" )
            save_img = (cropped.reshape(28,28) * 255).astype(np.uint8)
            cv2.imwrite(filename, save_img)
            st.success(f"Saved as {filename}")
        
    st.sidebar.header("Training")
    if st.sidebar.button("Retrain Model"):
        with st.spinner("Retraining model..."):
            subprocess.run(["python", "combine_train.py"], check=True)
        st.success("Model retrained")
        st.rerun()

    if cropped is not None:
        with st.expander("View Prediction Confidence"):
            fig, ax = pyplot.subplots()
            bars = ax.bar(range(10), prediction[0])
            bars[predicted_digit].set_color('red')
            ax.set_xlabel("Digit")
            ax.set_ylabel("Confidence")
            ax.set_title("Prediction Confidence")
            st.pyplot(fig)
    


