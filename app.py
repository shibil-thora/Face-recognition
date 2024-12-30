import cv2
import face_recognition
import numpy as np
import matplotlib.pyplot as plt

image_1 = "bean1.jpg"
image_2 = "bean2.jpg"

known_image = face_recognition.load_image_file(image_1)
known_encoding = face_recognition.face_encodings(known_image)[0]

new_face_image = face_recognition.load_image_file(image_2)
new_face_encodings = face_recognition.face_encodings(new_face_image)

def run():
    if new_face_encodings:
        matches = face_recognition.compare_faces(new_face_encodings, known_encoding)
        face_locations = face_recognition.face_locations(new_face_image)


        for match, (top, right, bottom, left) in zip(matches, face_locations):
            if match:
                print('Match found!')
                return
        print('No match found')
    else:
        print("No faces found in the new face image.") 

if __name__ == '__main__':
    run()