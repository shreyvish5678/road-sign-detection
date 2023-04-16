# import the opencv library
import cv2
# define a video capture object
vid = cv2.VideoCapture(0)
while(True):
  # Capture the video 
  ret, frame = vid.read()
  # Display the video
  cv2.imshow('video', frame)
  #Take a picture using 't'
  if cv2.waitKey(1) & 0xFF == ord('t'):
    #Initialize camera
    result, image = vid.read()
    #Show result and save and close with any button
    if result:
      cv2.imshow('picture', image)
      cv2.imwrite('input_image.png', image)
      cv2.waitKey(0)
      cv2.destroyWindow('picture')
    else:
      print('No picture') 
    cv2.destroyWindow('video')
    break
