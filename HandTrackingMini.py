import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # initializing a video capture object

mpHands = mp.solutions.hands  # creating instance of mediapipe
hands = mpHands.Hands()  # initializing the hand tracking model.
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime= 0

while True:  # infinite loop for capturing frame of the camera
    success, img = cap.read()  # captures frame and checks if successful
    if not success:
        print("Failed to capture frame.")
        break  # exit the loop if frame capture fails

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converts capture frame to RGB colors

    results = hands.process(imgRGB)  # detects hands using mediapipe

    # print(results.multi_hand_landmarks) #prints the landmarks of the detected hans

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id , lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                if id ==0:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (555, 0, 255), 3)
    cv2.imshow("Image", img)  # displays the captured frame

    if cv2.waitKey(1) & 0xFF == ord('q'):  # if q is pressed exits the programme
        break

cap.release()  # release the camera resources
cv2.destroyAllWindows()  # closes all the opencv windows
