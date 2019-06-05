import os

from dataset import load_batch

def eval_model(recognizer, meta_file):

    [list_img, list_label], num_sample = load_batch(meta_file)
    if num_sample < 1:
        return 0

    check = 0
    for i in range(num_sample):
        prediction, _ = recognizer.predict(list_img[i])

        check += prediction == list_label[i]

    return float(check) / num_sample

