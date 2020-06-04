import cv2
import sys
import argparse
import numpy as np
import os
import glob

def process_trajectories(trajectories_file_path):
    trajectories = dict()
    with open(trajectories_file_path) as dat:
        for row in dat:
            if len(row) > 20:
                row_split = row.rsplit()
                row_to_add = "0 " + row_split[4] + " " + row_split[2] + " " + row_split[5] + " " + row_split[3] # label, x1, y1, x2, y2
                if len(row_split) > 10:
                    frame_no = int(row_split[0])
                    if frame_no in trajectories:
                        trajectories[frame_no].append(row_to_add)
                    else:
                        trajectories[frame_no] = list()
                        trajectories[frame_no].append(row_to_add)

    return trajectories

if __name__ == '__main__':
    ''' Pretpostavka je da su sve slike/okviri istih dimenzija'''
    parser = argparse.ArgumentParser()

    parser.add_argument('--is_video', type=bool, default=False, help='set True if video is set as input')
    parser.add_argument('--video_file', type=str, default='t7.mp4', help='path to video file')
    parser.add_argument('--image_folder', type=str, default='Images', help='path to images')
    parser.add_argument('--trajectories', type=str, default='t7_trajektorije_2.txt', help='path to trajectories')
    parser.add_argument('--save_directory_image', type=str, default='Cropped images', help='place to save cropped images')
    parser.add_argument('--save_directory_labels', type=str, default='Cropped labels', help='place to save cropped labels')
    parser.add_argument('--num_of_parts', type=int, default=6, help='number of image crops')
    parser.add_argument('--from_frame', type=int, default=0, help='where to start')
    parser.add_argument('--to_frame', type=int, default=0, help='where to stop (inclusive)')
    parser.add_argument('--percentage', type=float, default=0.06, help='precentage of overlap between cropped images')

    opt = parser.parse_args()

    video = opt.is_video
    if video:
        video_file = opt.video_file
    else:
        images_folder = opt.image_folder

    trajectories_file_path = opt.trajectories
    save_dir_img = opt.save_directory_image

    if not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img)
    else:
        files = glob.glob(save_dir_img + "/*.*")
        for f in files:
            os.remove(f)

    save_dir_labels = opt.save_directory_labels

    if not os.path.exists(save_dir_labels):
        os.makedirs(save_dir_labels)
    else:
        files = glob.glob(save_dir_labels + "/*.*")
        for f in files:
            os.remove(f)

    num_parts = opt.num_of_parts
    start = opt.from_frame
    stop = opt.to_frame
    overlap_per = opt.percentage

    trajectories = process_trajectories(trajectories_file_path) # returns dictionary where frame number is key and value is list of player boxes (every mark is string)

    if video:
        cap = cv2.VideoCapture(video_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        offset = start
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start-1) # 0-based index
        width_part = int(np.ceil(width/num_parts))
        overlap = int(np.ceil(overlap_per * width))

        for i in range(start, stop+1):
            ret, frame = cap.read()
            x1 = 0
            x2 = width_part + overlap if (width_part + overlap) <= width else width_part
            for j in range(num_parts):
                crop_img = frame[0:(height+1), x1:(x2+1)].copy()
                cv2.imwrite(save_dir_img + "/frame" + str(i) + "_" + str(j) + ".png", crop_img)
                detections_for_frame = trajectories[i]
                for detection in detections_for_frame:
                    detection_split = detection.rsplit()
                    min_col = int(detection_split[1])
                    max_col = int(detection_split[3])
                    if (min_col >= x1) and (max_col <= x2):
                        min_col = min_col - x1
                        max_col = max_col - x1
                        exist_file = True
                        if not os.path.exists(save_dir_labels + "/frame" + str(i) + "_" + str(j) + ".txt"):
                            exist_file = False
                        with open(save_dir_labels + "/frame" + str(i) + "_" + str(j) + ".txt", "a+") as fil:
                            if not exist_file:
                                fil.write("0 " + str(min_col) + " " + detection_split[2] + " " + str(max_col) + " " + detection_split[4])
                            else:
                                fil.write("\n")
                                fil.write("0 " + str(min_col) + " " + detection_split[2] + " " + str(max_col) + " " + detection_split[4])
                x1 = x2 - overlap
                x2 = x1 + width_part + overlap if (x1 + width_part + overlap) <= width else width

    else:
        frame = cv2.imread(images_folder + "/frame" + str(start) + ".png")
        height, width = frame.shape[:2]
        width_part = int(np.ceil(width/num_parts))
        overlap = int(np.ceil(overlap_per * width))

        for i in range(start, stop+1):
            frame = cv2.imread(images_folder + "/frame" + str(i) + ".png")
            x1 = 0
            x2 = width_part + overlap if (width_part + overlap) <= width else width_part
            for j in range(num_parts):
                crop_img = frame[0:(height+1), x1:(x2+1)].copy()
                cv2.imwrite(save_dir_img + "/frame" + str(i) + "_" + str(j) + ".png", crop_img)
                detections_for_frame = trajectories[i]
                for detection in detections_for_frame:
                    detection_split = detection.rsplit()
                    min_col = int(detection_split[1])
                    max_col = int(detection_split[3])
                    if (min_col >= x1) and (max_col <= x2):
                        min_col = min_col - x1
                        max_col = max_col - x1
                        exist_file = True
                        if  not os.path.exists(save_dir_labels + "/frame" + str(i) + "_" + str(j) + ".txt"):
                            exist_file = False
                        with open(save_dir_labels + "/frame" + str(i) + "_" + str(j) + ".txt", "a+") as fil:
                            if not exist_file:
                                fil.write("0 " + str(min_col) + " " + detection_split[2] + " " + str(max_col) + " " + detection_split[4])
                            else:
                                fil.write("\n")
                                fil.write("0 " + str(min_col) + " " + detection_split[2] + " " + str(max_col) + " " + detection_split[4])
                x1 = x2 - overlap
                x2 = x1 + width_part + overlap if (x1 + width_part + overlap) <= width else width
