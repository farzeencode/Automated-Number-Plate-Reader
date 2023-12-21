import matplotlib.pyplot as plt
import cv2
import easyocr
from IPython.display import Image

harcascade = "haarcascade_russian_plate_number.xml"

cap = cv2.VideoCapture(0)
min_area = 500
count = 0

# width height
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()

    plate_cascade = cv2.CascadeClassifier(harcascade)
    # harcascade only accepts grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # to get plate coordinates
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
    print(plates)
    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            cv2.rectangle(img, (x, y), (x+w, y+h), color=(0, 18, 0), thickness=2)
            cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
            # to crop
            img_crop = img[y: y+h, x:x+w]
            cv2.imshow("crop", img_crop)

    cv2.imshow("result", img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        # save image to a file using imwrite()

        cv2.imwrite("plates_folder\plate_scanned"+str(count)+".jpg", img_crop)
        cv2.rectangle(img, (0, 200), (500, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "plate Saved", (150, 265), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1
        # Assuming output contains the text content
        Reader = easyocr.Reader(['en'])
        output = Reader.readtext(r"plates_folder\plate_scanned"+str(count)+".jpg")

        # Extracted text handling
        output_text = output[0][1]
        output_file_path = "output.txt"

        # Open the file in write mode
        with open(output_file_path, "w") as file:
            # Write the text content to the file
            file.write(output_text)

        # Provide a confirmation message
        print(f"Extracted text saved to {output_file_path}")
