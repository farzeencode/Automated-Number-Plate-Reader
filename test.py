import cv2

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
    # haarcascade only accept grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # to get plate co _ordinates
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)# Detects number plates in the grayscale 
    print(plates)
    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 18, 0), thickness=2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
            # to crop
            img_crop = img[y: y + h, x:x + w]
            cv2.imshow("crop", img_crop)

    cv2.imshow("result", img)

    key = cv2.waitKey(1)
    if key == 27:  # 27 corresponds to the Escape key
        break
    elif key == ord('s'):
        # save image to a folder using imwrite()
        cv2.imwrite("plates_folder/plate_scanned" + str(count) + ".jpg", img_crop)
        cv2.rectangle(img, (0, 200), (500, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1

cap.release()
cv2.destroyAllWindows()
