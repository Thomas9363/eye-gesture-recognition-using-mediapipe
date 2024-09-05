# eye-gesture-recognition-using-mediapipe
<img src="/introduction.GIF" alt="prototype" height="200">

## **hardware and software:**
laptop 16GB Surface Pro with Windows 11, Anaconda and PyCharm. Other modules are TensorFlow 2.14, OpenCV 4.8.1, MediaPipe 0.10.8, NumPy 1.26.4, and Jupyter Notebook.

Raspberry Pi 4GB Ram with Debian 12 Bookworm OS. Other modules are TensorFlow 2.16.1, OpenCV 4.9.0, MediaPipe 0.10.9, and NumPy 1.26.4.

## **Note:**
If you have no robotic eyes, you can run computer simulation ***eye_control_ball.py*** or ***iris_detect_tflite_ball.py***.

If you have a pair of robotic eyes, you can run ***eye_control_servo.py*** or ***iris_detect_tflite_servo.py***.
The training data ‘***iris_gesture_data.csv***’ is generated using my eyes. If you find not accurate, you may want to generate your own data.

## **Files:**
- eye_control_ball.py – initial attempt, moving a ball on screen
- eye_control_servo.py – initial attempt, control robotic eyes
- iris_creat_csv.py – generate training data
- iris_gesture_data.csv – training data for generating model
- iris_train.ipynb - model training script for eye gesture recognition
- iris_gesture_model.h5 – generated model in *.h5 format
- iris_gesture_model.tflite – generated model in *.tflite format
- eye_image.png – required image in simulation
- iris_image.png – required image in simulation	
- iris_detect_tflite_ball.py – inference script in computer simulation
- iris_detect_tflite_servo.py – inference script used on Pi to control eyes

## **Procedures:**
The link of the detailed instructions is at the bottom.

To begin the training, you set up classes for your eye movements from 0 to 9 and associated name. You need to remember what each number represents, as they will be used in the detection scripts.

 <img src="/eyeMove.png" alt="prototype" height="200">

- **Data Collection:**
 The script is ‘***iris_creat_csv.py***’. It uses MediaPipe face to detect and extract face landmarks in video frames. You position your face in front of the camera at slightly different angles. When ready, pressing keyboard keys 0 to 9, the x and y coordinates of the landmarks in the eye region are sorted in ‘***iris_gesture_data.csv***’.

 <img src="/dataCollection.jpg" alt="prototype" height="300">

- **Model Training:**
 Run Jupyter Notebook in PyCharm, select '***iris_train.ipynb***’ and run all. The ‘***iris_gesture_model.tflite***’ is the trained model that can be deployed on either Windows OS or a Raspberry Pi.

- **Model Deployment:**
 There are two detection scripts. The script ‘***iris_detect_tflite_ball.py***’ uses a graphical interface to test the accuracy of the data on laptop. This script displays a pair of eyes on the screen that follow the movement of your eyes. The script ‘***iris_detect_tflite_servo.py***’ is used to control the robotic eyes.

<img src="/simulation.GIF" alt="prototype" height="300">

## **License:**
Codes are under [Apache v2 license](https://github.com/Thomas9363/eye-gesture-recognition-using-mediapipe/blob/main/LICENSE).

The detailed instructions are at [my instructables](https://www.instructables.com/Control-Robotic-Eyes-With-My-Eyes-Using-AI-and-Dee/)
