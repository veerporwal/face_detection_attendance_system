import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("face-attendence-with-real-time-firebase-adminsdk-fw8ua-7cd123b24c.json")
firebase_admin.initialize_app(cred,
                              {'databaseURL': 'https://face-attendence-with-real-time-default-rtdb.firebaseio.com/',
                               'storageBucket': 'face-attendence-with-real-time.appspot.com'})

#extracting student images
folderPath = 'Images'
studentid = []

if folderPath is not None:  
    modePathList = os.listdir(folderPath)
    studentList = []
    
    for path in modePathList:
        studentList.append(cv2.imread(os.path.join(folderPath, path)))
        studentid.append(os.path.splitext(path)[0])
        fileName = f'{folderPath}/{path}'
        bucket = storage.bucket()
        blob = bucket.blob(fileName)
        blob.upload_from_filename(fileName)
    
    print(len(studentList))
    print(studentid)
else:
    print("Error: Unable to read image from 'Images'")

def find_encoding(imageList):
    encodeList = []
    for img in imageList:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        
    print(studentid)
    return encodeList
    

print("Encoding Started ...")
encoding_known = find_encoding(studentList)
encodeknownwithid = [encoding_known, studentid]
print(encoding_known)
print("Encoding Completed ...")
file = open('Encodefile.p', 'wb')
pickle.dump(encodeknownwithid, file)
file.close()
print("File saved!")

