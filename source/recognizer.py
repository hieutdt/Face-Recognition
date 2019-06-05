import os
import cv2
import sys
import config
from dataset import load_img, preprocess

def recognize(img_file, uid_file):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    model_filename = os.path.join(config.OUTPUT_DIR,config.OUTPUT_MODEL_FILE)
    recognizer.read(model_filename)

    faceCascade = cv2.CascadeClassifier(config.CASCADE_PATH)

    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    with open(uid_file) as f:
        uids = [line.strip() for line in f.readlines()]

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x + w, y + h), (0,255,0), 3)
        img_predict = gray[y:y+h, x:x+w]
        img_predict = preprocess(img_predict)
        img_predict = cv2.resize(img_predict, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        Id, score = recognizer.predict(img_predict)
        if score > 0:
            cv2.putText(img, uids[Id], (x, y), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 3)
            print("%s" %(uids[Id]))
        else:
            cv2.putText(img, "unknown", (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    cv2.imwrite("detect.jpg", img)
    

if __name__ == "__main__":
    if len(sys.argv) == 3:
        recognize(sys.argv[1], sys.argv[2])