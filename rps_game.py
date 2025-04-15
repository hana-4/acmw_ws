from model import RPSModule
loaded_model = RPSModule()
# Instantiate a new instance of our model (this will be instantiated with random weights)
import requests
from pathlib import Path
import torch
import torch
import cv2
import numpy as np

# Define the URL and local file path
url = "https://github.com/hana-4/acmw_ws/raw/main/rps_model_weights.pth"
save_dir = Path("./weights")
save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
weights_path = save_dir / "rps_model_weights.pth"

# Download the weights file if it doesn't exist
if not weights_path.exists():
    print(f"Downloading weights from {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(weights_path, "wb") as f:
            f.write(response.content)
        print(f"Weights downloaded and saved to {weights_path}")
    else:
        print(f"Failed to download weights: {response.status_code}")
else:
    print(f"Weights file already exists at {weights_path}")

# Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
loaded_model.load_state_dict(torch.load(f=weights_path))


# Load your PyTorch model
model = loaded_model# Replace with your actual loaded model
model.eval()  # Set the model to evaluation mode

# Get the expected input size from the model
input_model_size = (128, 128)  # Update this based on your model's input size

# Define class labels
class_labels = {0: 'paper', 1: 'rock', 2: 'scissors'}

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow('Rock Paper Scissors')

# Define ROIs for Player 1 and Player 2
roi1_x, roi1_y, roi1_w, roi1_h = 200, 150, 300, 300 # ROI for Player 1
roi2_x, roi2_y, roi2_w, roi2_h = 800, 150, 300, 300  # ROI for Player 2

def determine_winner(player1_choice, player2_choice):
    if player1_choice == player2_choice:
        return "Draw"
    elif (player1_choice == 'rock' and player2_choice == 'scissors') or \
         (player1_choice == 'scissors' and player2_choice == 'paper') or \
         (player1_choice == 'paper' and player2_choice == 'rock'):
        return "Player 1 Wins!"
    else:
        return "Player 2 Wins!"

while True:
    try:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Extract ROIs for both players
        roi1 = frame[roi1_y:roi1_y+roi1_h, roi1_x:roi1_x+roi1_w]
        roi2 = frame[roi2_y:roi2_y+roi2_h, roi2_x:roi2_x+roi2_w]

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        def predict_choice(roi):
            """Predicts the class label for a given ROI."""
            # Convert ROI to grayscale, resize, normalize, and convert to tensor
            roi_resized = cv2.resize(roi, input_model_size, interpolation=cv2.INTER_AREA)
            roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            roi_normalized = roi_gray / 255.0
            roi_tensor = torch.tensor(roi_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Classify the ROI image using the PyTorch model
            with torch.no_grad():
                prediction_prob =(model(roi_tensor)).squeeze(0).cpu().numpy()
                #prediction_prob=model(roi_tensor)q

            predicted_class_index = np.argmax(prediction_prob)
            predicted_class_label = class_labels[predicted_class_index]
            predicted_class_prob = prediction_prob[predicted_class_index]

            if predicted_class_prob < 0.9:
                return 'Undefined'
            else:
                return predicted_class_label


        # Predict choices for both players
        player1_choice = predict_choice(roi1)
        player2_choice = predict_choice(roi2)

        # Determine winner
        if player1_choice != 'Undefined' and player2_choice != 'Undefined':
            result = determine_winner(player1_choice, player2_choice)
        else:
            result = "Waiting for valid input..."

        # Display ROIs and predictions
        cv2.rectangle(frame, (roi1_x, roi1_y), (roi1_x + roi1_w, roi1_y + roi1_h), (255, 0, 0), 2)
        cv2.putText(frame, f'P1: {player1_choice}', (roi1_x, roi1_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.rectangle(frame, (roi2_x, roi2_y), (roi2_x + roi2_w, roi2_y + roi2_h), (0, 255, 0), 2)
        cv2.putText(frame, f'P2: {player2_choice}', (roi2_x, roi2_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the result
        cv2.putText(frame, result, (400, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Show the frame
        cv2.imshow('Rock Paper Scissors', frame)

    except Exception as e:
        print(e)
        break

cap.release()
cv2.destroyAllWindows()
