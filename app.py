from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model
import serial
import serial.tools.list_ports
import platform
import time

app = Flask(__name__)

# Global variables for UART
ser = None
last_port = None

# UART Functions
def list_available_ports():
    """List all available serial ports on the system"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"Found port: {port.device}")
        print(f"Description: {port.description}")
        print(f"Hardware ID: {port.hwid}")
        print("---")
    return ports

def find_silicon_labs_port():
    """Find the Silicon Labs CP210x port"""
    ports = list_available_ports()
    for port in ports:
        if "CP210x" in port.description:
            return port.device
    return None

def get_port_name():
    """Get the appropriate port name based on the operating system"""
    system = platform.system()
    
    if system == "Windows":
        return 'COM1'
    elif system == "Linux":
        return '/dev/ttyUSB0'
    else:
        return None

def initialize_serial():
    global ser
    try:
        if ser is not None and ser.is_open:
            ser.close()
            ser = None
            
        # Try to find the correct port
        port = find_silicon_labs_port() or get_port_name()
        if not port:
            print("UART Debug: No suitable port found")
            return False
            
        print(f"UART Debug: Attempting to connect to {port}...")
        ser = serial.Serial(
            port=port,
            baudrate=9600,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1,
            write_timeout=1
        )
        
        if ser.is_open:
            print(f"UART Debug: Successfully connected to {port}")
            return True
        return False
            
    except serial.SerialException as e:
        print(f"UART Debug: Serial Error: {e}")
        return False
    except Exception as e:
        print(f"UART Debug: Unexpected error: {e}")
        return False

def send_emotion_to_serial(emotion):
    global ser
    try:
        if not ser or not ser.is_open:
            print("UART Debug: Attempting to reconnect...")
            if not initialize_serial():
                print("UART Debug: Could not establish connection")
                return
            
        emotion_commands = {
            'Happy': '2',
            'Sad': '1'
        }
        
        if emotion in emotion_commands:
            command = emotion_commands[emotion]
            print(f"UART Debug: Sending command {command} for {emotion}")
            
            ser.write(command.encode('ascii'))
            ser.flush()
            print(f"UART Debug: Command {command} sent successfully")
            
            time.sleep(0.1)
            if ser.in_waiting:
                response = ser.read(ser.in_waiting)
                response_str = response.decode('ascii', errors='ignore').strip()
                print(f"UART Debug: Received response: {response_str}")
                
    except Exception as e:
        print(f"UART Debug: Error in send_to_serial: {e}")

# Load the pre-trained model
model = load_model('my_model.h5')

# Load Haar Cascade for face detection
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Dictionary for emotion labels - only happy and sad
labels_dict = {
    3: 'Happy',
    5: 'Sad'
}

# Function to capture video and detect emotion
def gen_frames():
    video = cv2.VideoCapture(0)

    while True:  # ✅ No indentation error here
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)
        
        for x, y, w, h in faces:
            sub_face_img = gray[y:y+h, x:x+w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]

            # Only process if the emotion is happy (3) or sad (5)
            if label in [3, 5]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Print emotion to terminal and send to UART
                print(f"Detected Emotion: {labels_dict[label]}")
                send_emotion_to_serial(labels_dict[label])

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

    video.release()

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/main')
def main():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Cleanup for serial connection when the app closes
@app.teardown_appcontext
def cleanup(error):
    global ser
    if ser is not None and ser.is_open:
        ser.close()
        ser = None

if __name__ == '__main__':
    print("UART Debug: Starting application...")
    print(f"Running on: {platform.system()}")
    
    # List all available ports
    print("Available ports:")
    list_available_ports()
    
    # Initialize serial connection
    retry_count = 3
    while retry_count > 0:
        if initialize_serial():
            print("UART Debug: Initial serial connection successful")
            break
        else:
            print(f"UART Debug: Failed to establish initial serial connection, retries left: {retry_count-1}")
            retry_count -= 1
            time.sleep(1)
    
    if not ser or not ser.is_open:
        print("Warning: Starting without serial connection. Will try to connect when needed.")
    
    app.run(debug=True, use_reloader=False)
