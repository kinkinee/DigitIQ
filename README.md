# DigitIQ
A CNN-based handwritten digit recognition web application built using TensorFlow, OpenCV, and Streamlit.
DigitIQ can classify handwritten digits from both uploaded images and a real-time drawing canvas. The application includes confidence visualization, user feedback collection, and model retraining capabilities to continuously improve prediction accuracy.

---

## Features

- Handwritten digit recognition (0–9)

- Upload digit images for prediction

- Draw digits directly on a canvas

- Real-time prediction confidence visualization

- User feedback collection for incorrect predictions

- Retraining with custom handwritten datasets

- Custom dataset support alongside MNIST

---

## Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit
- Matplotlib
- Pillow

---

## Project Structure

```text
DigitIQ/
├── dr_cnn.py
├── combine_train.py
├── requirements.txt
├── README.md
├── digit_cnn_combined.keras

```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/kinkinee/DigitIQ.git
cd DigitIQ
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run dr_cnn.py
```

or

```bash
py -3.11 -m streamlit run dr_cnn.py
```

The application will launch in your browser.

---

## Dataset

The model is trained using:

- MNIST handwritten digit dataset
- Custom handwritten digit dataset

Place your custom dataset in the following format:

```text
my_digits/
├── 0/
├── 1/
├── 2/
├── 3/
├── 4/
├── 5/
├── 6/
├── 7/
├── 8/
└── 9/
```

Each folder should contain images corresponding to that digit.

> Note: The dataset is not included in this repository due to size constraints.

---

## Model Training

To train or retrain the model:

```bash
python combine_train.py
```

The trained model will be saved as:

```text
digit_cnn_combined.keras
```

---

## Feedback-Based Learning

If a prediction is incorrect:

1. Select the correct digit.
2. Submit feedback.
3. The image is added to the custom dataset.
4. Retrain the model to incorporate the new samples.

This allows the system to gradually improve on handwriting styles not present in the original training data.

---

## Example Workflow

```text
User Input
    ↓
Image Preprocessing
    ↓
CNN Prediction
    ↓
Confidence Visualization
    ↓
Feedback Collection
    ↓
Dataset Expansion
    ↓
Model Retraining
```

---

## Future Improvements

- Multi-digit recognition
- Phone number recognition
- Character recognition (A–Z, a–z)
- OCR-style text extraction
- Cloud deployment
- User authentication and prediction history

---

## Author

Developed by Kinkinee.


---

## License

This project is intended for educational and learning purposes.
