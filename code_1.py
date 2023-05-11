import cv2
import pytesseract
import numpy as np
import sys

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

cascade = cv2.CascadeClassifier("C:\\Users\\navee\\Desktop\\capstone\\Number_Plate_Detection\\haarcascade_russian_plate_number.xml")

states = {
    "AP9CD6789": "NAVEEN UUSA -FLAT NO.2204",
    "MH02BY3123": "HARSHAL JAIN -FLAT NO.5203",
    "HR10AF3753": "BALAJI AVINASH -FLAT NO.4342"
}

def extract_num(img_name):
    img = cv2.imread(img_name)
    if img is None:
        print("Failed to read image:", img_name)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in nplate:
        a, b = (int(0.02 * img.shape[0]), int(0.025 * img.shape[1]))
        plate = img[y + a:y + h - a, x + b:x + w - b, :]
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        _, plate_gray = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)

        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())

        if read in states:
            print("Car belongs to:", states[read])
        else:
            print("UNKNOWN VEHICLE")
        print("License Plate:", read)

        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), (51, 51, 255), -1)
        font_scale = 1.0
        thickness = 2
        color = (255, 255, 255)

        cv2.putText(img, read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    cv2.imshow("License Plate Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Check if the image path is provided as a command-line argument
if len(sys.argv) < 2:
    print("Please provide the image path as a command-line argument.")
    sys.exit(1)
# Get the path of the uploaded image from the command-line argument
image_path = sys.argv[1]
print(image_path)

# Call the function to extract the number plate
extract_num(image_path)
