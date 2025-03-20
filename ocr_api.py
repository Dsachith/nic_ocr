import cv2
import pytesseract
import re
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

def detect_face(image):
    """
    Detect faces in the image using OpenCV's Haar Cascade.
    Returns True if a face is detected, False otherwise.
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.2,
        minNeighbors=10,
        minSize=(50, 50)
    )
    
    return len(faces) > 0 

def validate_nic(text):
    """
    Validate NIC format from extracted text.
    """
    nic_pattern_old = r'\b\d{9}[VX]\b'
    nic_pattern_new = r'\b\d{12}\b'
    
    match_old = re.search(nic_pattern_old, text)
    match_new = re.search(nic_pattern_new, text)

    if match_old:
        return {"valid": True, "nic": match_old.group(), "format": "Old NIC"}
    elif match_new:
        return {"valid": True, "nic": match_new.group(), "format": "New NIC"}
    else:
        return {"valid": False, "message": "Invalid NIC"}

def process_image(image_path):
    """
    Process image for face detection and NIC extraction.
    """
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face_detected = detect_face(gray)

    gray = cv2.equalizeHist(gray)

    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    gray = cv2.medianBlur(gray, 3)

    text = pytesseract.image_to_string(gray, lang="eng")

    print("Extracted Text:--------------", text)  

    result = validate_nic(text)
    result["face_detected"] = face_detected  
    return result

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_path = "uploaded_id.jpg"
    file.save(image_path)

    result = process_image(image_path)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

