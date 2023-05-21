import cv2
import numpy as np

lines = []
count = 0

def mm(event, x, y, flags, params):
    global count
    # while(count < 4):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        lines.append((x, y))
        print(x, y)
        count+=1 

img = cv2.imread("1.png")

while True:
    cv2.imshow("im", img)
    cv2.setMouseCallback("im", mm)
    print(count)
    cv2.waitKey(0)

i = 0
# while True:
print(lines)
while i+1 < len(lines):
    t = lines[i]
    b = lines[i+1]
    i += 2
    cv2.line(img, t, b, (0,255,0), 4)
cv2.imshow("image", img)
cv2.waitKey(1)