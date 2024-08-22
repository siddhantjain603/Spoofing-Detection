# Spoofing Detection

## Overview
The Spoofing Detection Project aims to differentiate between real and fake faces using a combination of face detection and deep learning models. The project consists of several scripts to collect data, train models, and test the system.

## Project Structure
The project is organized into the following key files:

- `dataCollection.py`: Collects and processes data from the webcam, detects faces, and saves labeled images and bounding box coordinates.
- `main.py`: Runs a real-time spoofing detection using a YOLO model.
- `train.py`: Trains a YOLO model on the collected dataset.
- `splitdata.py`: Splits the dataset into training, validation, and test sets.
- `camTest.py`: Provides a simple script to test the webcam feed.
- `faceDetector.py`: A script to test face detection functionality.
- `yoloTest.py`: Tests the YOLO model with the webcam feed.

## Installation
To get started with the Spoofing Detection Project, follow these steps:

1. **Clone the repository:**
    ```sh
    git clone https://github.com/siddhantjain603/Spoofing-Detection.git
    cd Spoofing-Detection
    ```

2. **Set up a virtual environment (optional but recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install required packages:**
    Create a `requirements.txt` file with the following content:
    ```txt
    mediapipe
    ultralytics
    cvzone
    ```
    Then install the packages using:
    ```sh
    pip install -r requirements.txt
    ```

4. **Download YOLO model weights:**
    Download the pre-trained YOLO model weights and place them in the `models` directory.

## File Descriptions

### `dataCollection.py`
- **Purpose:** Collects data from the webcam and labels it as real or fake based on face detection and blurriness.
- **Dependencies:** `cv2`, `cvzone`, `time`
- **Configuration:**
  - `classID`: 0 for fake, 1 for real
  - `outputFolderPath`: Path to save collected images and labels
  - `confidence`: Minimum confidence score for detecting faces
  - `blurThreshold`: Threshold to determine image blurriness

### `main.py`
- **Purpose:** Real-time spoofing detection using a YOLO model.
- **Dependencies:** `cv2`, `cvzone`, `ultralytics`
- **Configuration:**
  - `confidence`: Minimum confidence score for detections

### `train.py`
- **Purpose:** Trains a YOLO model on the dataset.
- **Dependencies:** `ultralytics`
- **Configuration:**
  - `epochs`: Number of training epochs

### `splitdata.py`
- **Purpose:** Splits the dataset into training, validation, and test sets.
- **Dependencies:** `os`, `random`, `shutil`
- **Configuration:**
  - `splitRatio`: Ratio for splitting data into train, validation, and test sets

### `camTest.py`
- **Purpose:** Provides a simple script to view the webcam feed.
- **Dependencies:** `cv2`

### `faceDetector.py`
- **Purpose:** Tests face detection functionality using a webcam.
- **Dependencies:** `cv2`, `cvzone`
- **Configuration:**
  - `minDetectionCon`: Minimum detection confidence for face detection

### `yoloTest.py`
- **Purpose:** Tests the YOLO model with the webcam feed.
- **Dependencies:** `cv2`, `cvzone`, `ultralytics`
- **Configuration:**
  - `model`: Path to YOLO model weights

## Usage

1. **Collect Data:**
   Run `dataCollection.py` to collect images and bounding box coordinates from the webcam.

2. **Train Model:**
   Use `train.py` to train the YOLO model on the collected dataset.

3. **Test Model:**
   Run `main.py` to test the trained model with real-time video from the webcam.

4. **Test Webcam Feed:**
   Use `camTest.py` to view the live feed from the webcam.

5. **Test Face Detection:**
   Run `faceDetector.py` to verify face detection functionality.

6. **Test YOLO Model:**
   Use `yoloTest.py` to see YOLO model predictions on the webcam feed.

## Notes

- Make sure to adjust the file paths and model weights as needed for your specific setup.
- You might need to modify configuration parameters in the scripts to better fit your environment and use case.
