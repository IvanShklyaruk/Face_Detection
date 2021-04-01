import cv2

# Loaded data on frontal faces.
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Chosen image to detect faces in
img = cv2.imread('RDJ.png')

# Convert the image to greyscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# Draw the rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Show the image with the faces
cv2.imshow('Ivan Shklyaruk Image Face Detection', img)
cv2.waitKey()