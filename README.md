# Emotion-recognition-app
Developed a machine learning model to recognize human emotions from text inputs. Utilized Natural Language Processing techniques and trained classifiers to categorize emotions like joy, anger, sadness, fear, and surprise. Deployed using Streamlit for an interactive user interface.

The Facial Recognition App is a lightweight yet powerful application designed to accurately identify and verify individuals using their facial features. Developed with Flask, this web-based application leverages the capabilities of TensorFlow and Keras for deep learning, enabling real-time face detection and recognition. With OpenCV handling image and video processing, the app can capture faces from live camera feeds or uploaded images. The front-end is seamlessly served using Flask routes, while Matplotlib and Pillow assist in visualizing and managing images effectively during training and testing phases.

This application incorporates robust machine learning workflows using Scikit-Learn for preprocessing and classification tasks, backed by Numpy for efficient numerical computations. The facial recognition model is trained on facial image datasets and is capable of distinguishing between multiple users with high accuracy. Whether used for secure logins, attendance systems, or identity verification, the app showcases the practical potential of integrating AI with simple web interfaces to solve real-world problems.

## features
1) Real-time face detection using webcam
2) Emotion detection capability
3) Responsive UI built with Flask templates
4) Integrated visualization for training results
5) User-friendly interface with live feedback

## User Interface Preview

| Home/Login Page | Live Detection | Result Screen |
|-----------------|----------------|----------------|
| ![UI Screenshot 1](https://github.com/profitter261/Emotion-recognition-app/blob/main/User%20interface%20images/Screenshot%202025-03-17%20111548.png) | ![UI Screenshot 2](https://github.com/profitter261/Emotion-recognition-app/blob/main/User%20interface%20images/Screenshot%202025-03-17%20111812.png) | ![UI Screenshot 3](https://github.com/profitter261/Emotion-recognition-app/blob/main/User%20interface%20images/Screenshot%202025-04-09%20213210.png) |

## Requirements

1) Flask==2.2.3
2) Werkzeug==2.2.3
3) tensorflow==2.11.0
4) keras==2.11.0
5) numpy==1.24.2
6) opencv-python==4.7.0.72
7) matplotlib==3.6.3
8) pillow==9.4.0
9) scikit-learn==1.2.1

## procedure/working

1) Data Collection
2) Capture face images using OpenCV and categorize them by user identity.
3) Store images in folders labeled by user names for easy training.
4) Model Development
5) Use TensorFlow and Keras to build a Convolutional Neural Network (CNN).
6) Train the model on the collected facial dataset for recognition accuracy.
7) Model Saving and Loading
8) Save the trained model in .h5 format.
9) Load the model into the Flask app to make predictions in real-time.
10) Flask App Integration
11) Develop a Flask backend to control routes and render the UI.
12) Use OpenCV with Flask to access the webcam feed for face detection.
13) Real-Time Recognition
14) Recognize faces from live webcam feed.
15) Display recognition results dynamically on the web interface.
16) Visualization
17) Plot training/validation accuracy and loss using Matplotlib.
18) Use Pillow for processing and displaying images.

## Outcomes

Developed a fully functional real-time facial recognition web application.
Integrated deep learning (TensorFlow) with OpenCV and Flask.
Achieved over 90% recognition accuracy with custom-trained CNN.
Enabled webcam-based live face recognition and UI display.
Extended the application with emotion detection (optional).

## Conclusion

This project demonstrates the power of combining computer vision and web development to build interactive and intelligent systems. By integrating machine learning with a lightweight web framework, the facial recognition app provides a scalable and user-friendly solution for authentication and security use-cases. It also sets the foundation for further enhancements like cloud deployment, database support, and multi-user management.






