# EmoDetect

A real-time emotion detection application built with TensorFlow and OpenCV. EmoDetect uses a Convolutional Neural Network (CNN) to classify facial expressions into predefined emotion categories such as Happy, Sad, Angry, Surprise, Neutral, Fear, and Disgust. The application can process static images as well as real-time webcam feeds.

---

## Project Overview

EmoDetect combines OpenCV's Haar Cascade face detector with a CNN model trained on facial emotion data. Faces are detected in images or video streams, preprocessed, and passed through the CNN to output the most likely emotion.

---

## Project Structure

```plaintext
EmoDetect/
├── main.py                                # Emotion detection script for images or webcam
├── test.py                                # Script to test on sample images
├── testdata.py                            # Utilities for test data handling
├── haarcascade_frontalface_default.xml    # OpenCV Haar Cascade face detector
├── model_file.h5                          # Trained CNN model weights
├── requirement.txt                        # Python dependencies
├── .gitignore                             # Files and folders to ignore in Git
├── download.jpeg                          # Sample images for demonstration
├── emotion.jpeg
├── happy.jpeg
├── sad.jpeg
├── profile.jpg
└── README.md                              # Project documentation (this file)
```

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/grimreapermanasvi/EmoDetect.git
   cd EmoDetect
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirement.txt
   ```

---

## Usage

### Real-time Webcam Detection

```bash
python main.py --webcam 0 --model model_file.h5
```

- **--webcam**: Index of the webcam device (default: 0)  
- **--model**: Path to the trained model weights

### Static Image Emotion Detection

```bash
python main.py --image path/to/image.jpg --model model_file.h5
```

- **--image**: Path to the input image file  
- **--model**: Path to the trained model weights

### Running Tests on Sample Images

```bash
python testdata.py --data_folder . --model model_file.h5
```

- **--data_folder**: Folder containing sample images (e.g., download.jpeg)  
- **--model**: Path to the trained model weights

---

## Dataset

EmoDetect uses a CNN model trained on facial expression datasets (e.g., FER-2013). If you'd like to retrain or fine-tune the model, prepare your data in folders by emotion and use a training script of your choice.

---

## Model Architecture

The CNN model consists of multiple Conv2D and MaxPooling2D layers with ReLU activations, interleaved with Dropout for regularization, followed by Dense layers culminating in a softmax output for classification.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

*Happy coding!*
