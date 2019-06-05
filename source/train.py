import config
import sys
import numpy as np
import datetime
import os
import cv2
from dataset import load_batch
from eval import eval_model
import init_dataset_meta as idm

def train(meta_files):
    if len(meta_files) < 3:
        meta_files = idm.init_dataset_meta()
    if not os.path.exists(config.OUTPUT_DIR):
        os.mkdir(config.OUTPUT_DIR)

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    #global_accuracy = 0
    
    [list_img, list_label], num_sample = load_batch(meta_files[0])
    if num_sample < 1:
        print('Err: 0 sample found')

    recognizer.train(list_img,np.array(list_label))
    #recognizer.update(list_img, np.array(list_label))

    train_accuracy = eval_model(recognizer, meta_files[0])
    val_accuracy = eval_model(recognizer, meta_files[1])
    test_accuracy = eval_model(recognizer, meta_files[2])

    print('Train accuracy =  %f' %(train_accuracy))
    print('Test accuracy = %f' %(test_accuracy))
    print('Validate accuracy = %f' %(val_accuracy))

    recognizer.write(os.path.join(config.OUTPUT_DIR, config.OUTPUT_MODEL_FILE))

if __name__ == "__main__":
    if len(sys.argv) == 4:
        train(sys.argv[1:])
    else:
        train([])
