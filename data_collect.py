import numpy as np
import cv2
import FACE_DET

cap=cv2.VideoCapture(0)
pause = True
counter = 2500
while True:
            ret,frame = cap.read()
            faces = FACE_DET.detect_faces(frame.copy())
            for x1,y1,x2,y2 in faces:
                        roi = frame.copy()[y1:y2,x1:x2]
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.imshow("Faces",frame)
            if not pause:
                        cv2.imwrite("Corona Mask/Mask/"+str(counter)+".jpg",roi)
                        counter+=1
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                        break
            if k == ord("p"):
                        pause = not(pause)
            print(counter)
            if counter == 2510:
                        break

cap.release()
cv2.destroyAllWindows()
