import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from time import time

####################################################################
classID = 0 # Use 1 for real person and 0 for fake
outputFolderPath = 'Dataset/DataCollect'
confidence = 0.8
save = True
blurThreshold = 50

debug = False
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640,480
floatingPoint = 6
###############################################################

cap = cv2.VideoCapture(0)
cap.set(3,camWidth)
cap.set(4,camHeight)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

while True:
    success, img = cap.read()
    imgOut = img.copy()
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = [] # True False values indicating image is blur or not
    listInfo = [] # Normalised values and the class name for label text file

    if bboxs:
        for bbox in bboxs:
            score = float(bbox["score"][0])
            x, y, w, h = bbox['bbox']

            # Check the score
            if score> confidence:

                # Calculate offset in pixels
                offsetW = (offsetPercentageW / 100) * w
                offsetH = (offsetPercentageH / 100) * h

                # Adjust the bounding box coordinates
                x -= int(offsetW)  # Move x left by offsetW
                y -= int(offsetH * 2)  # Move y up by 2*offsetH
                w += int(offsetW * 2)  # Increase width by offsetW on both sides
                h += int(offsetH * 2.5)  # Increase height by 2*offsetH

                # T avoid value below 0
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                # Finding blurriness
                imgFace = img[y:y+h,x:x+w]
                cv2.imshow("Face",imgFace)
                blur_value = int(cv2.Laplacian(imgFace,cv2.CV_64F).var())

                if blur_value > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # Normalizing the values
                ih,iw,_ = img.shape
                cx,cy = x+w/2, y+h/2
                cxNormalised, cyNormalised = round(cx/iw,floatingPoint),round(cy/ih,floatingPoint)
                wNormalised, hNormalised = round(w/iw,floatingPoint),round(h/ih,floatingPoint)

                # To avoid values above 1
                if cxNormalised > 1 : cxNormalised = 1
                if cyNormalised > 1: cyNormalised = 1
                if wNormalised > 1: wNormalised = 1
                if hNormalised > 1: hNormalised = 1

                listInfo.append(f'{classID} {cxNormalised} {cyNormalised} {wNormalised} {hNormalised}\n')


                # Drawings
                cvzone.cornerRect(imgOut, (x, y, w, h))
                cvzone.putTextRect(imgOut, f'Score: {int(score*100)}% Blur: {blur_value}',(x,y-15),scale=2, thickness = 2)

                if debug:
                    cvzone.cornerRect(imgOut, (x, y, w, h))
                    cvzone.putTextRect(imgOut, f'Score: {int(score*100)}% Blur: {blur_value}',(x,y-15),scale=2, thickness = 2)

        # To save
        if save:
            if all(listBlur) and listBlur!=[]:

                # Save Image
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0] + timeNow[1]
                cv2.imwrite(f'{outputFolderPath}/{timeNow}.jpg',img)

                # Save Label Text File
                for info in listInfo:
                    f = open(f'{outputFolderPath}/{timeNow}.txt', 'a')
                    f.write(info)
                    f.close()

    cv2.imshow("Image", imgOut)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
