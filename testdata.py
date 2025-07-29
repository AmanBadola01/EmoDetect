#######Original code #########
# import cv2
# import numpy as np
# from keras.models import load_model
# import random
# import os

# model=load_model('model_file.h5')

# faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

# # len(number_of_image), image_height, image_width, channel
# ##Use for a single image 
# image_path = input("Enter the full path to the image file: ").strip()

# # Read the image
# frame = cv2.imread(image_path)

# # # Path to test directory
# # test_dir = r"C:\Users\MANASVI\Documents\GitHub\Emotion_detection\test"

# # # Step 1: Get list of emotion folders
# # emotion_folders = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]

# # # Step 2: Choose a random folder
# # chosen_folder = random.choice(emotion_folders)

# # # Step 3: Get list of image files in that folder
# # image_files = [f for f in os.listdir(chosen_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# # # Step 4: Pick a random image
# # chosen_image_path = os.path.join(chosen_folder, random.choice(image_files))

# # print(f"Using image: {chosen_image_path}")

# # # Load and process the image
# # frame = cv2.imread(chosen_image_path)
# if frame is None:
#     print("Could not read the image. Please check the path.")
# else:
#     gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces= faceDetect.detectMultiScale(gray, 1.3, 3)
#     for x,y,w,h in faces:
#         sub_face_img=gray[y:y+h, x:x+w]
#         resized=cv2.resize(sub_face_img,(48,48))
#         normalize=resized/255.0
#         reshaped=np.reshape(normalize, (1, 48, 48, 1))
#         result=model.predict(reshaped)
#         label=np.argmax(result, axis=1)[0]
#         print(label)
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
#         cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
#         cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
#     cv2.imshow("Frame",frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


#####Modified code ######
import cv2
import numpy as np
from keras.models import load_model
import os
import random

# Load the trained model and face detector
model = load_model('model_file.h5')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Label dictionary
labels_dict = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Neutral', 5: 'Sad', 6: 'Surprise'
}

# Ask user for image path
image_path = input("Enter image path (or press Enter to use random test image): ").strip()

if not image_path:
    test_dir = r"C:\Users\MANASVI\Documents\GitHub\Emotion_detection\test"
    folders = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]
    random_folder = random.choice(folders)
    images = [f for f in os.listdir(random_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    image_path = os.path.join(random_folder, random.choice(images))
    print(f"[INFO] Using random test image: {image_path}")

# Read image
frame = cv2.imread(image_path)

if frame is None:
    print("[ERROR] Could not load image. Check path.")
    exit()

# Resize for better viewing without distortion
scale_percent = 150  # 150% of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

print(f"[INFO] Faces detected: {len(faces)}")

if len(faces) == 0:
    print("[WARNING] No faces found!")
else:
    for x, y, w, h in faces:
    # Predict the emotion
        face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = labels_dict[np.argmax(result)]

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # Draw label *above* the face rectangle
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        label_y = max(y - 10, label_size[1] + 10)
        cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print(f"[INFO] Detected emotion: {label}")

# Show result
cv2.imshow("Emotion Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()