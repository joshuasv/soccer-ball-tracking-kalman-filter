import os
import argparse

import cv2

def video2frames(fname):
    dir_path = f'frames-{fname}'
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    cap = cv2.VideoCapture(fname)
    if cap.isOpened() == False:
        print('Error while opening the video.')
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            cv2.imwrite(f'{dir_path}/{count}.jpg', frame)
            count += 1
        else:
            break
    cap.release()
    print(f'All frames extracted to {dir_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', metavar='V', type=str, nargs=1, help='video path')

    args = parser.parse_args()

    video2frames(args.video[0])

if __name__ == '__main__':
    main()
