import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from datetime import datetime

cred = credentials.Certificate("face-attendence-with-real-time-firebase-adminsdk-fw8ua-7cd123b24c.json")
firebase_admin.initialize_app(cred,
                              {'databaseURL': 'https://face-attendence-with-real-time-default-rtdb.firebaseio.com/',
                               'storageBucket': 'face-attendence-with-real-time.appspot.com'})

bucket = storage.bucket()
#web cam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

#graphics 
imgBackground = cv2.imread('Resources\\background.png')
folderModePath = 'Resources\\Modes'
#import images into a list
if folderModePath is not None:  
    modePathList = os.listdir(folderModePath)
    imgModeList = []
    
    for path in modePathList:
        imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
    
    print(len(imgModeList))
else:
    print("Error: Unable to read image from 'Resources\\Modes'")

#loading the encoded file
print("Loading the encoded file ...")
file = open('Encodefile.p', 'rb')
encodeknownwithid = pickle.load(file)
file.close()
encodeknown, studentid = encodeknownwithid
print("File loaded!")

modeType = 0
counter = 0
imgstudent = []
while True:
    success, img = cap.read()
    
    imgs = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    faceCurFace = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs, faceCurFace)
    
    imgBackground[162:162+480, 55:55+640] = img
    imgBackground[44:44+633, 808: 808+414] = imgModeList[modeType]

    if faceCurFace:
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFace):
            matches = face_recognition.compare_faces(encodeknown, encodeFace)
            faceDis = face_recognition.face_distance(encodeknown, encodeFace)
            print("faceDis",faceDis)
            print("matches", matches)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                # print("Known face detected!")
                # print(studentid[matchIndex])
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                id = studentid[matchIndex]
                if counter == 0:
                    counter = 1
                    modeType = 1
            if counter != 0:

                if counter == 1:
                    studentinfo = db.reference(f'students/{id}').get()
                    print(studentinfo)
                    #Get the images from the storage
                    blob = bucket.blob(f'Images/{id}.png')
                    array = np.frombuffer(blob.download_as_string(), np.uint8)
                    imgstudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                    #update the attendence
                    ref = db.reference(f'students/{id}')
                    studentinfo['total_attendance'] += 1
                    #update date of attendance
                    datetimeObject = datetime.strptime(studentinfo['Last_attendance'], "%Y-%m-%d %H:%M:%S")
                    secondElapsed = (datetime.now()-datetimeObject).total_seconds()
                    print(secondElapsed)
                    if secondElapsed > 30:
                        ref.child('total_attendance').set(studentinfo['total_attendance'])
                        ref.child('Last_attendance').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    else:
                        modeType = 3
                        counter = 0
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if modeType != 3:
                    if 10 < counter < 20:
                        modeType = 2

                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                    if counter <= 10:
                        cv2.putText(imgBackground, str(studentinfo['total_attendance']), (861, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentinfo['Branch']), (1006, 550),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(id), (1006, 493),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentinfo['semester']), (910, 625),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                        cv2.putText(imgBackground, str(studentinfo['current_year']), (1025, 625),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                        cv2.putText(imgBackground, str(studentinfo['starting_year']), (1125, 625),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
                        
                        (w, h), _ = cv2.getTextSize(studentinfo['Name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                        offset = (414 - w) // 2
                        cv2.putText(imgBackground, str(studentinfo['Name']), (808 + offset, 445),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)
                    
                    imgstudent_resized = cv2.resize(imgstudent, (216, 216))

                    imgBackground[175: 175 + 216, 909 : 909 + 216] = imgstudent_resized
                    counter += 1
                    
                    if counter >= 20:
                            counter = 0
                            modeType = 0
                            studentInfo = []
                            imgStudent = []
                            imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    else:
        modeType = 0
        counter = 0
        #imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    cv2.imshow("Face Attendence", imgBackground)
    
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Break the loop if 'q' is pressed
        break

