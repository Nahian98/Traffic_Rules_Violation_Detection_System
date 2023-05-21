import cv2
import numpy as np

img = cv2.imread("3.png")
blur = cv2.GaussianBlur(img, (5,5), 0)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
low_white = np.array([195,195,195])
up_white = np.array([200,200,200])
mask = cv2.inRange(img, low_white, up_white)
edges = cv2.Canny(mask, 75, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=10)

for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img, (x1,y1), (x2,y2), (0,255,0), 2)



cv2.imshow("image", img)
# cv2.imshow("hsv", hsv)
# cv2.imshow("mask", mask)
# cv2.imshow("edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()