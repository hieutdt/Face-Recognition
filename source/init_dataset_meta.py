import glob
import random
import os
import numpy as np
import sys

import config

def init_dataset_meta(data_dir=config.DATASET_DIR, 
        ptrain=config.PTRAIN, pvalidate=config.PVAL, ptest=config.PTEST):
    if not os.path.exists(data_dir):
        print('Err...')
        return([])

    counts = np.zeros(config.NUM_CLASS) #la hang so tuong ung so luong
    paths = [] #mang mot chieu luu tat ca duong link
    class_folders = glob.glob(data_dir + '/*') #tim tat ca file trong thu muc data_dir
    class_folders.sort() #sort cho dung thu tu

    for ii in range(config.NUM_CLASS):
        iipaths = glob.glob(class_folders[ii]  + '/*')
        random.shuffle(iipaths)
        counts[ii] = len(iipaths) #so luong anh cua nguoi do
        paths.append(iipaths)

    train_paths =[]
    test_paths = []
    validate_paths = []

    for ii in range(config.NUM_CLASS):
        num_train = int (ptrain * counts[ii]) #phan tram nhan so anh
        num_test = int(ptest * counts[ii])
        #num_valid = counts[ii] - num_train - num_test

        spl_paths = paths[ii]
        train_paths += spl_paths[0:num_train]
        test_paths += spl_paths[num_train:num_train + num_test]
        validate_paths += spl_paths[num_train + num_test:]


    random.shuffle(train_paths)
    random.shuffle(test_paths)
    # ramdom.shuffle(validate_paths)

    filenames = ['./train.txt', './validate.txt' , './test.txt']
    allpaths = [train_paths, validate_paths, test_paths]

    for ii in range(3):
        fn = filenames[ii]
        iipaths = allpaths[ii]
        
        f = open(fn, 'w')
        for jj in iipaths:
            f.write(os.path.abspath(jj) + '\t' + str(class_folders.index(os.path.dirname(jj)))+ '\n')
        f.close()
    f = open('./uid.txt', 'w')
    for jj in class_folders:
        f.write(os.path.basename(jj) + '\n')
    f.close()



if __name__ =='__main__':
    init_dataset_meta()
