import cv2

width = 800
height = 600
blue = (255, 0, 0)

# load the image, resize it, and convert it to grayscale
image = cv2.imread("images/1.jpg")
image = cv2.resize(image, (width, height))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the haar cascades face and smile detectors
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
smile_detector = cv2.CascadeClassifier("haarcascade_smile.xml")

# detect faces in the grayscale image
face_rects = face_detector.detectMultiScale(gray, 1.1, 8)

# loop over the face bounding boxes
for (x, y, w, h) in face_rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), blue, 2)
    # extract the face from the grayscale image
    roi = gray[y:y + h, x:x + w]

    # apply the smile detector to the face roi
    smile_rects, rejectLevels, levelWeights = smile_detector.detectMultiScale3(
                                                        roi, 2.5, 20, outputRejectLevels=True)

    # if there was no detection, we consider this a "no smiling" detection
    if len(levelWeights) == 0:
        cv2.putText(image, "Not Smiling", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, blue, 3)
    else:
        # if `levelWeights` is below 2, we classify this as "Not Smiling"
        if max(levelWeights) < 2:
            cv2.putText(image, "Not Smiling", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, blue, 3)
        # otherwise, there is a smiling in the face ROI
        else:
            cv2.putText(image, "Smiling", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, blue, 3)

cv2.imshow("image", image)
cv2.waitKey(0)
