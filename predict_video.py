import cv2 as cv
import numpy as np
import urllib.request
from predict import predict_eye
from PIL import Image

urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt', 'deploy.prototxt')
urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel', 'res10_300x300_ssd_iter_140000.caffemodel')

net = cv.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

weight="/content/drive/MyDrive/weights.pth"

video_capture=cv.VideoCapture("/content/drive/MyDrive/eyetest2.mp4")

video_codec=cv.VideoWriter_fourcc(*'mp4v')


width = 640
height = 480
fps = video_capture.get(cv.CAP_PROP_FPS)
print(fps)



if fps==0:
  fps=20.0

output=cv.VideoWriter("Face_detected.mp4", video_codec, fps, (width,height))

while True:
  ret,frame=video_capture.read()

  if not ret:
    break

  frame=cv.resize(frame,(width,height)) 
  blob=cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

  net.setInput(blob)

  detected_values=net.forward()
  (h,w)=frame.shape[:2]

  for i in range(detected_values.shape[2]):
    high_value=detected_values[0,0,i,2]

    if high_value>0.6: #if face has  more than 60% accuracy
      box=detected_values[0, 0, i, 3:7] * np.array([w, h, w, h])
    
      (startx,starty,endx,endy)=box.astype("int")


      face_crop=frame[starty:endy,startx:endx]

      face_rgb=cv.cvtColor(face_crop,cv.COLOR_BGR2RGB)
      result=predict_eye(face_rgb,weight)
      left_eye=result[2]
      eye_center = (int(left_eye[0] + (startx-15)),int(left_eye[1] + (starty+15)))
      cv.ellipse(frame, center=eye_center, axes=(20, 10), angle=0, startAngle=0, endAngle=360, color=(255, 0, 0), thickness=2)




  output.write(frame)


video_capture.release()
output.release()
print("process completed")
