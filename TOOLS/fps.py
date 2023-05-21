import cv2
import os
# Opens the Video file
cap= cv2.VideoCapture('/media/ubuntu/Transcend/NVR/2022-4-19/NVR_ch2_main_20220419080000_20220419090000.dav')
i=1
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i%75 == 0:
        cv2.imwrite(os.path.join('/media/ubuntu/Transcend/Extracted Frame', 'demo'+str(i)+'.jpg'),frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()