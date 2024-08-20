# Importing modules
import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
import csv
import datetime

print("Opening Application..")
print("Please stand in front of the camera")

# Capturing video
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread("C:\\Attendance System\\Files\\Resources\\background.png")

# Importing the mode images into a list
"C:\Attendance System\Files\Resources\Modes"
folderModePath = "C:\Attendance System\Files\Resources\Modes"
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))


# Load the encoding file
print("Loading Encode File ...")
file = open("C:\Attendance System\Files\EncodeFile.p", 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded")

# Download student details:
file = open("C:\Attendance System\student details.csv", 'r')
csv_reader = csv.reader(file, delimiter='\t')
student_details = {row[0].split(",")[0]: row[0].split(",") for row in csv_reader}
file.close()

modeType = 0
counter = 0
id = -1

# Students who are present:
present_students = {}


while True:
    success, img = cap.read()

    # Resizing and converting images:
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Face encodings and face locations
    # Main
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162: 162+480, 55:55+640] = img
    imgBackground[44: 44+633, 808:808+414] = imgModeList[modeType]

    # Matches

    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                # Main
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                #print("matches", matches)
                #print("faceDis", faceDis)

                matchIndex = np.argmin(faceDis)
                # print("Match Index", matchIndex)

                if matches[matchIndex]:
                    # print("Known Face Detected")
                    #print(studentIds[matchIndex])

                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                    imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                    id = studentIds[matchIndex]

                    if counter == 0:
                        counter = 1
                        modeType = 1
                
        if counter != 0:
                if counter == 1:
                    print("Student ID: ", id)
                    if id not in present_students:
                        time = datetime.datetime.now()
                        string_time = time.strftime('%Y-%m-%d %H:%M:%S')
                        present_students[id] = [id, student_details[id][1], student_details[id][2], int(student_details[id][3]) + 1, str(string_time)]     
                        f = open("C:\\Attendance System\\attendance.csv", "a", newline = "")
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(present_students[id])
                        f.close()
                    
                    else:
                        modeType = 3
                        counter = 0
                        imgBackground[44: 44+633, 808:808+414] = imgModeList[modeType]

                if modeType != 3:

                    if 150 < counter < 160:
                        modeType = 2

                    imgBackground[44: 44+633, 808:808+414] = imgModeList[modeType]

                    if counter <= 150:    
                        # Showing the correct details of the student on the screen
                        cv2.putText(imgBackground, student_details[id][3], (861, 125),
                                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                        cv2.putText(imgBackground, student_details[id][2], (1006, 550),
                                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground,student_details[id][0], (1006, 493),
                                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                        (w, h), _ = cv2.getTextSize(student_details[id][1], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                        offset = (414 - w) // 2
                        cv2.putText(imgBackground, student_details[id][1], (808 + offset, 445),
                                                cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
                        
                        # Getting the image of the student:
                        student_img = cv2.imread(student_details[id][4])
                        student_img_resize = cv2.resize(student_img, (216, 216))
                        imgBackground[175:175 + 216, 909:909 + 216] = student_img_resize
                
                    counter +=1

                    if counter>= 160:
                        counter = 0
                        modeType = 0
                        imgBackground[44: 44+633, 808:808+414] = imgModeList[modeType]
    
    else:
        modeType = 0
        counter = 0
                    

    #cv2.imshow('Webcam', img)
    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
