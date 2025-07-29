
# Emotion Detection from Images

This project detects human emotions from static image input using deep learning and computer vision. It uses OpenCV for face detection and a pre-trained CNN model to predict the emotion.

## 🧠 Emotions Detected

- Angry 😠  
- Disgust 😖  
- Fear 😨  
- Happy 😄  
- Sad 😢  
- Surprise 😲  
- Neutral 😐  

## 🛠️ Technologies Used

- Python
- OpenCV
- Keras / TensorFlow
- NumPy
- Matplotlib (for visualization)
- Jupyter Notebook

## 📁 Folder Structure

```
Emotion_detection/
├── emotion_model.h5        # Trained CNN model
├── test_image.png          # Sample image for testing
├── test_code.ipynb         # Notebook to run emotion detection
├── emotion_env/            # Virtual environment (ignored by git)
├── .gitignore              # Ignores emotion_env/ from Git tracking
└── README.md               # This file
```

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/grimreapermanasvi/EmotionDetection.git
   cd EmotionDetection
   ```

2. **Create and Activate Virtual Environment**
   ```bash
   python -m venv emotion_env
   emotion_env\Scripts\activate  # Windows
   ```

3. **Install Requirements**
   *(You can generate `requirements.txt` by running `pip freeze > requirements.txt` first)*
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Notebook**
   Open `test_code.ipynb` and run all cells. Upload an image and detect the emotion.

## 🧪 Sample Output

The model draws a bounding box around the detected face and displays the predicted emotion above it.

## 📌 Notes

- Make sure `emotion_env/` is added to `.gitignore`.
- The model file `emotion_model.h5` should be kept under 100MB to be pushed to GitHub, or uploaded using Git LFS.

