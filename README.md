# Number-Plate-Detection
How to detect car license plate
import cv2

plat_detector = cv2.CascadeClassifier(r"D:\Datascience Classes\12th may\Computer vision tutorial master\Haarcascades\haarcascade_license_plate_rus_16stages.xml")
video = cv2.VideoCapture(r"D:\Datascience Classes\12th may\internship practical\car2.mp4")

if not video.isOpened():
    print('Error reading video')

while True:
    ret, frame = video.read()
    
    if not ret:
        break

    gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plat_detector.detectMultiScale(gray_video, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    
    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        frame[y:y + h, x:x + w] = cv2.blur(frame[y:y + h, x:x + w], ksize=(10, 10))
        cv2.putText(frame, text='License Plate', org=(x - 3, y - 3), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    color=(0, 0, 255), thickness=1, fontScale=0.6)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
