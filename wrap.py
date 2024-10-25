import cv2
import numpy as np

img=cv2.imread("images/cards.jpg")
p1=np.float32([[77,150],[261,150],[76,444],[263,444 ]])
width,height=250,350
p2=np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix= cv2.getPerspectiveTransform(p1,p2)
out=cv2.warpPerspective(img,matrix,(width,height))
for i in range(0,4):
    cv2.circle(img, (int(p1[i][0]), int(p1[i][1])), 5, (0, 0, 255), cv2.FILLED)
cv2.imshow("image",img)
cv2.imshow("out",out)
cv2.waitKey(0)
cv2.destroyAllWindows()