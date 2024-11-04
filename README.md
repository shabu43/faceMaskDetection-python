
# Face Mask Detection

This project implements a face mask detection system using deep learning with Keras and TensorFlow. The model is trained to classify images as containing faces with or without masks, utilizing the MobileNetV2 architecture. The system is designed for real-time mask detection using a webcam feed and will provide an audio alert when a person is detected without a mask.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Real-Time Detection](#real-time-detection)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features
- Real-time face mask detection using a webcam.
- Uses a pre-trained MobileNetV2 model for image classification.
- Provides audio alerts for detected individuals without masks.
- Outputs a classification report with precision, recall, and F1 scores.

## Installation

To set up the environment for this project, make sure you have Python installed (preferably 3.6 or later). Then, you can install the required packages using pip. You can copy and paste the following commands in your terminal:

```bash
pip install --upgrade tensorflow keras sklearn imutils opencv-python scipy numpy pygame
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Face-Mask-Detection.git
   cd Face-Mask-Detection
   ```

2. Prepare your dataset:
   - Organize your dataset in the following structure:
     ```
     dataset/
     ├── with_mask/
     │   ├── image1.jpg
     │   ├── image2.jpg
     │   └── ...
     └── without_mask/
         ├── image1.jpg
         ├── image2.jpg
         └── ...
     ```
   - Replace the `DIRECTORY` variable in the training script with the path to your dataset.

3. Train the model:
   - Run the training script (`train_mask_detector.py`):
     ```bash
     python train_mask_detector.py
     ```

4. Start real-time detection:
   - After training, run the detection script (`detect_mask.py`):
     ```bash
     python detect_mask.py
     ```

## Model Training

The training process uses Keras with TensorFlow as the backend. It applies data augmentation techniques to enhance the robustness of the model. The training process is completed in 20 epochs with a batch size of 32.

The classification report is saved as `report.txt` and includes metrics such as precision, recall, and F1-score.

### Example of classification report:
```
              precision    recall  f1-score   support

   with_mask       0.93      0.79      0.85       198
without_mask       0.86      0.96      0.91       272

    accuracy                           0.89       470
   macro avg       0.90      0.87      0.88       470
weighted avg       0.89      0.89      0.89       470
```

## Real-Time Detection

The detection script utilizes OpenCV for real-time video processing. It captures video from the default camera, detects faces, predicts whether they are wearing masks, and displays the results on the screen.

### Audio Alerts

If an individual is detected without a mask, an audio alert is played. Ensure you have an audio file named `Recording.mp3` in the same directory.

## License

This project is licensed under the MIT License. See the LICENSE file for more information.

## Acknowledgments

- The dataset can be sourced from various public repositories or created by collecting images in the specified format.
- This project utilizes the following libraries:
  - TensorFlow
  - Keras
  - OpenCV
  - Scikit-learn
  - Imutils
  - Pygame
