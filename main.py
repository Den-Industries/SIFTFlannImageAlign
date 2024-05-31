import os
import glob
import cv2 as cv
import numpy as np
import datetime

res_val = 2

# Путь к папке с изображениями
input_folder = 'D:/Documents/prikol/prikol/plan1/*'
# Путь к папке для сохранения стыкованных изображений
output_folder = 'D:/Documents/prikol/prikol/plan1ob/'

# Считываем все имена файлов из папки
filenames = glob.glob(input_folder)
filenames.sort()

successfule = []

computed_earler = []

for file_index in range(len(filenames)):
    s = list(filenames[file_index])
    s[42] = '/'
    filenames[file_index] = "".join(s)

MIN_MATCH_COUNT = 140
FLANN_INDEX_KDTREE = 1

GOOD_COUNT = 0

sift = cv.SIFT_create()
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

def compute_descriptors(img):
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def match_img(descriptors_template, descriptors_frame):
    matches = flann.knnMatch(descriptors_template, descriptors_frame, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)

    global GOOD_COUNT
    GOOD_COUNT = len(good)
    if len(good) > MIN_MATCH_COUNT:
        pts_img = np.float32([kp_template[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_frame = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        homography_matrix, homography_mask = cv.findHomography(pts_img, pts_frame, cv.RANSAC, 5.0)
        if np.any(homography_matrix):
            return homography_matrix
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    return None


def draw_replacement_on_frame(frame, pts_on_frame, homography_matrix,template_size):
    perspective = cv.perspectiveTransform(pts_on_frame, homography_matrix)
    perspective_matrix = cv.getPerspectiveTransform(pts_on_frame, perspective)

    # set img_sub to template location and perspective
    img_res = cv.resize(img_template, (img_template.shape[1], img_template.shape[0]))

    out_sub = cv.warpPerspective(img_res, perspective_matrix, (frame.shape[1], frame.shape[0]), borderValue=[0, 0, 0])

    mask = np.where(out_sub != [0, 0, 0])
    frame[mask] = out_sub[mask]

    # add template points to frame
    #for pts in perspective:
        #frame = cv.circle(frame, (int(pts[0][0]), int(pts[0][1])), 3, (0, 0, 255), 1)

    return frame


if __name__ == "__main__":
    print(filenames[0])
    successfule.append(filenames[0])
    file_index = 1
    #for file_index in range(1, len(filenames)):
    while file_index < len(filenames):
        print(file_index)
        outputed_filename = os.path.join(output_folder, os.path.basename(filenames[file_index]))
        if os.path.isfile(outputed_filename) and not (outputed_filename in successfule):
            successfule.append(outputed_filename)
        file_index += 1
    file_index = 1
    while file_index < len(filenames):
        print(file_index)
        outputed_filename = os.path.join(output_folder, os.path.basename(filenames[file_index]))
        if os.path.isfile(outputed_filename):
            if not (outputed_filename in successfule):
                successfule.append(outputed_filename)
            file_index += 1
            continue

        print(filenames[file_index])
        buf1 = cv.imread(filenames[file_index])
        if buf1 is None:
            continue
        buf1 = cv.resize(buf1, (int(4608 / res_val), int(3456 / res_val)),
                          interpolation=cv.INTER_AREA)

        img_template = buf1
        kp_template, descriptors_template = compute_descriptors(img_template)

        for sucfile in successfule:
            buf2 = cv.imread(sucfile)
            buf2 = cv.resize(buf2, (int(4608 / res_val), int(3456 / res_val)), interpolation=cv.INTER_AREA)

            # find img_template coordinates for perspective on frame
            template_size = buf2.shape
            pts_on_frame = np.float32(
                [[0, 0], [0, template_size[0] - 1], [template_size[1] - 1, template_size[0] - 1],
                 [template_size[1] - 1, 0]]).reshape(-1, 1, 2)

            start = datetime.datetime.now()

            frame = buf2
            kp_frame, descriptors_frame = 0, 0
            havebeen = False
            for compared in computed_earler:
                if compared[0] == sucfile:
                    kp_frame, descriptors_frame = compared[1]
                    havebeen = True
                    break
            if not havebeen:
                kp_frame, descriptors_frame = compute_descriptors(frame)
                computed_earler.append((sucfile, (kp_frame, descriptors_frame)))

            homography_matrix = match_img(descriptors_template, descriptors_frame)
            stop = datetime.datetime.now()
            print('Time', stop - start, (stop - start).microseconds)
            if np.any(homography_matrix):
                print("HOMO ", filenames[file_index], sucfile, GOOD_COUNT)
                frame = draw_replacement_on_frame(frame, pts_on_frame, homography_matrix, template_size)
                output_filename = os.path.join(output_folder, os.path.basename(filenames[file_index]))
                cv.imwrite(output_filename, frame)
                if GOOD_COUNT < 350:
                    successfule.append(output_filename)
                break
            else:
                print("NO HOMO", filenames[file_index], sucfile)
                if sucfile == successfule[-1]:
                    print("WILL HOMO LATER")
                    filenames.append(filenames[file_index])
        file_index += 1
