import cv2
vid = cv2.VideoCapture(2)
u=0
while u<1000:
 ret, frame=vid.read()
 u+=1
def get():
 ret, frame=vid.read()
 #cv2.imshow("",frame)
 #cv2.waitKey(1)
 #if frame is not None:
 #frame=cv2.resize(frame, (640, 480))
 #cv2.imwrite("image.jpg",frame)
 return frame