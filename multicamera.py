import cv2
cap1 = cv2.VideoCapture(0)


while True:
    ret1, frame1 = cap1.read()
    

    if ret1:
        cv2.imshow("cap1",frame1)

    

    if cv2.waitKey(1) == ord('q'):
        break
cap1.release()

cv2.destroyAllWindows()
