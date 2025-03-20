import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp  # For hand detection

# Load the trained gesture recognition model
model = load_model('asl_model.h5')

# Define the class labels (0-9 and A-Z)
class_labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Detect only one hand

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Define the fixed aspect ratio (400x400)
fixed_size = (400, 400)

# Define the scaling factor for the bounding box (increase to make the box larger)
scale_factor = 1.5  # Adjust this value as needed

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # If a hand is detected, crop and preprocess it
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the bounding box of the hand
            h, w, _ = frame.shape
            x_min = min([landmark.x for landmark in hand_landmarks.landmark]) * w
            x_max = max([landmark.x for landmark in hand_landmarks.landmark]) * w
            y_min = min([landmark.y for landmark in hand_landmarks.landmark]) * h
            y_max = max([landmark.y for landmark in hand_landmarks.landmark]) * h

            # Calculate the center of the bounding box
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2

            # Calculate the size of the bounding box
            box_width = x_max - x_min
            box_height = y_max - y_min

            # Use the larger dimension (width or height) to create a square bounding box
            box_size = max(box_width, box_height)

            # Scale the bounding box size
            box_size = int(box_size * scale_factor)

            # Calculate the new coordinates for the fixed-size bounding box
            x_min = int(center_x - box_size / 2)
            x_max = int(center_x + box_size / 2)
            y_min = int(center_y - box_size / 2)
            y_max = int(center_y + box_size / 2)

            # Ensure the bounding box stays within the frame boundaries
            x_min = max(0, x_min)
            x_max = min(w, x_max)
            y_min = max(0, y_min)
            y_max = min(h, y_max)

            # Crop the hand region
            hand_roi = frame[y_min:y_max, x_min:x_max]

            # Preprocess the hand region
            if hand_roi.size != 0:  # Ensure the ROI is not empty
                # Resize the hand region to match the model input size
                resized_roi = cv2.resize(hand_roi, fixed_size)
                normalized_roi = resized_roi / 255.0  # Normalize pixel values
                input_roi = np.expand_dims(normalized_roi, axis=0)  # Add batch dimension

                # Debugging: Visualize the hand ROI
                cv2.imshow('Hand ROI', resized_roi)

                # Make a prediction
                predictions = model.predict(input_roi)
                predicted_class = np.argmax(predictions, axis=1)[0]
                predicted_label = class_labels[predicted_class]

                # Display the predicted label on the frame
                cv2.putText(frame, f"Gesture: {predicted_label}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw the bounding box around the hand (optional)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Live Gesture Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()