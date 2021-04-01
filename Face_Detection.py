import cv2

# Loaded data on frontal faces.
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Chosen image to detect face in
img = cv2.imread('RDJ.png')

# Convert the image to greyscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw the rectangle around the face
(x, y, w, h) = face_coordinates[0]
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)











cv2.imshow('Ivan Shklyaruk Face Detection', img)
cv2.waitKey()

print("Code Completed!")