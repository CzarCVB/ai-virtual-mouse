import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

#####################
frameR = 100
hCam, wCam = 720, 1280
smoothening = 7
wScr, hScr = autopy.screen.size()
######################

cTime = 0
pTime = 0

plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
while True:
    success, img = cap.read()
    # Find hand landmarks
    img = detector.findHands(img, draw=False)
    lmlist, bbox = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        # get tip of index and middle
        x1, y1 = lmlist[8][1:]
        cv2.circle(img, (x1, y1), 8, (255, 0, 255))
        x2, y2 = lmlist[12][1:]
        # check which fingers are up
        fingers = detector.fingersUp()
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 255, 0), 2)
        # only index finger: moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # convert coordinates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            # smoothen values
            clocX = plocX + (x3-plocX)/smoothening
            clocY = plocY + (y3-plocY)/smoothening
            # move mouse
            autopy.mouse.move(wScr-clocX, clocY)
            cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        # two fingers up, clicking mode
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            # click mouse if distance short
            if length < 45:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 8, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 8, 255), 3)
    cv2.imshow("AI Gesture Controlled Mouse", img)
    cv2.waitKey(1)
