import cv2
import pytesseract
import re
import numpy as np

def validate_nic(text):
    nic_pattern_old = r'\b\d{9}[VX]\b'  # Old NIC format (e.g., 123456789V)
    nic_pattern_new = r'\b\d{12}\b'  # New NIC format (e.g., 200012345678)
    
    match_old = re.search(nic_pattern_old, text)
    match_new = re.search(nic_pattern_new, text)

    if match_old:
        return f"Valid Old NIC: {match_old.group()}"
    elif match_new:
        return f"Valid New NIC: {match_new.group()}"
    else:
        return "Invalid ID"

def process_image(image_path):
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to improve OCR accuracy
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Extract text using Tesseract OCR
    text = pytesseract.image_to_string(gray, lang="eng")
    print("Extracted Text:", text)

    # Validate NIC number
    validation_result = validate_nic(text)
    print(validation_result)

def capture_image():
    cap = cv2.VideoCapture(0)  # Open the default camera (0)
    
    while True:
        ret, frame = cap.read()
        cv2.imshow("Capture ID Card (Press 's' to save)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):  # Save image when 's' is pressed
            cv2.imwrite("id_card.jpg", frame)
            print("Image saved as id_card.jpg")
            break
        elif key == ord("q"):  # Quit without saving when 'q' is pressed
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
capture_image()  # Capture ID card image
process_image("id_card.jpg")  # Process and validate the captured image
