import cv2
import face_recognition
import pickle
import os

# importing student images
folderPath = "C:\Attendance System\Files\Images"
pathList = os.listdir(folderPath)
imgList = []
studentIds = []
for path in pathList:
    image = cv2.imread(os.path.join(folderPath, path))
    resized_image = cv2.resize(image, (216, 216))
    imgList.append(resized_image)
    studentIds.append(os.path.splitext(path)[0])


# Find encodings of the faces
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # main
        encode = face_recognition.face_encodings(img)[0]
        print(encode)
        encodeList.append(encode)

    return encodeList

print("Encoding started.....")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding complete")

"C:\Attendance System\Files\Images"
# Storing the encodings in a pickle file
file = open("C:\Attendance System\Files\EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")


















