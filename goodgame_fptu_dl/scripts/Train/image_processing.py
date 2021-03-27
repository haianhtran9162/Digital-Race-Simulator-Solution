import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
import common


# TAT CA CAC HAM DUNG XU LY ANH SE DUOC LUU TAI DAY

# roi to get driving view
def getDrivingView(img, skyline):
    width = img.shape[1]
    height = img.shape[0]

    return img[skyline:height, 0:width]


def convertBGRtoGray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def convertBGRtoHSV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def convertBGRtoHLS(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)


def adaptiveThresh(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 0)


def adaptiveThreshInverse(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 0)


def thresh(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)


def inRange1Channel(img, min_threshold, max_threshold):
    return cv2.inRange(img, np.array(min_threshold), np.array(max_threshold))


def inRange3Channels(img, min_threshold, max_threshold):
    return cv2.inRange(img, np.array(min_threshold), np.array(max_threshold))


def canny(img, first_threshold, second_threshold):
    return cv2.Canny(img, first_threshold, second_threshold)


def sobel(img):
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)


def findBiggestContour(img):
    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    maxCntArea = 0
    indexMax = 0

    for i, cnt in enumerate(contours):
        cntArea = cv2.contourArea(cnt)
        if (cntArea > maxCntArea):
            maxCntArea = cntArea
            indexMax = i

    result = np.zeros_like(img)

    cv2.drawContours(result, contours, indexMax, (255), -1)
    return result


def erode(image, kernel_shape):
    return cv2.erode(image, np.ones(kernel_shape, dtype=np.uint8), iterations=1)


def dilate(image, kernel_shape):
    return cv2.dilate(image, np.ones(kernel_shape, dtype=np.uint8), iterations=1)


# ham copy nguyen si tu code FPT
def birdViewTransform(img, birdview_config):
    width = img.shape[1]  # column
    height = img.shape[0]  # row

    birdView_width = 240
    birdView_height = 380

    src_vertices = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    dst_vertices = np.array([[0, 0], [birdView_width, 0],
                             [birdView_width - 105, birdView_height], [105, birdView_height]], np.float32)

    M = cv2.getPerspectiveTransform(src_vertices, dst_vertices)

    return cv2.warpPerspective(img, M, (birdView_width, birdView_height), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)


def getHistogram(one_channel_img):
    return cv2.calcHist([one_channel_img], [0], None, [256], [0, 256])


def getFFTHistogram(histogram):
    histogram = histogram.T.squeeze()
    rft = np.fft.rfft(histogram)
    rft[21:] = 0
    return np.fft.irfft(rft)


def localMinimumOnFFTHistogram(fft_histogram):
    return np.array([
        (color - 1)
        for color, val in enumerate(fft_histogram)
        if
        (color > 1) &
        (val > fft_histogram[color - 1]) & (fft_histogram[color - 1] < fft_histogram[color - 2])
    ])


# dam bao local_minimums da duoc sort asc
def get2LocalMinimumsSurround(local_minimums, color):
    result = [
        (local_minimums[i], local_minimums[i + 1])
        for i in range(len(local_minimums) - 1)
        if (local_minimums[i] <= color) and (local_minimums[i + 1] >= color)
    ]
    if (result is None or len(result) == 0):
        return (0, 0)
    else:
        return result[len(result) - 1]


# lay cac gia tri trong khoang cac cuc tieu, bao gom
# mau toi nhat, mau sang nhat, mau xuat hien nhieu nhat,
# dien tich cua dai mau nay
def getColorStripsOfFFTHistogram(histogram, fft_histogram):
    local_minimums = localMinimumOnFFTHistogram(fft_histogram)
    color_strips_number = len(local_minimums) - 1
    color_strips = np.zeros((color_strips_number, 5), dtype=np.int)

    for i in range(len(color_strips)):
        lower_color = local_minimums[i]
        upper_color = local_minimums[i + 1]

        color_strips[i][0] = lower_color
        color_strips[i][2] = upper_color
        color_strips[i][1] = np.argmax(histogram[lower_color:upper_color]) + lower_color
        color_strips[i][3] = histogram[color_strips[i][1]]
        color_strips[i][4] = np.sum(histogram[lower_color:upper_color + 1], dtype=np.int)

    return color_strips


def getLargeAreaColorStrips(histogram, fft_histogram):
    color_strips = getColorStripsOfFFTHistogram(histogram, fft_histogram)
    total_area = np.sum(histogram, dtype=np.int)

    return color_strips[(color_strips[:, 4] > total_area // 10) & (
                color_strips[:, 3] > color_strips[:, 4] // (color_strips[:, 2] - color_strips[:, 0]))]


def findOtherArea(gray_image, lower_color, upper_color):
    inrange_image = inRange1Channel(gray_image, lower_color, upper_color)

    return inrange_image


# fill all road by whitening mid-lane
def fillRoad(birdview_image):
    height = birdview_image.shape[0]
    width = birdview_image.shape[1]

    bottom_row = birdview_image[height - 1, :]
    left_col = (bottom_row != 0).argmax()
    right_col = width - (np.flip(bottom_row) != 0).argmax()
    birdview_image[height - 1, left_col:right_col] = 255

    biggest_cnt = findBiggestContour(birdview_image)

    return biggest_cnt


def calculateDistanceCarToLane(lane, point):
    distances = np.sqrt(np.sum((lane - point) ** 2, axis=1))
    return distances


def normalizeDistance(distances, config_data):
    d_max = np.array(config_data["d_max"], dtype=np.float)
    d_min = np.array(config_data["d_min"], dtype=np.float)
    return (distances - d_min) / (d_max - d_min)


def normalizeDistanceByMaxDistance(distances, d_max):
    normalize_d = distances / np.array(d_max, dtype=np.float)
    normalize_d[normalize_d > 1] = 1
    return normalize_d


# tra ve true neu diem (x, y) nam ben trai cua centerLine, neu ben phai tra ve false
def isLeftOfCenterLine(x, y, center_line):
    # by cross product
    x0 = center_line[0][0]
    y0 = center_line[0][1]

    x1 = center_line[1][0]
    y1 = center_line[1][1]

    sign = np.sign((x0 - x1) * (y - y1) - (y0 - y1) * (x - x1))
    return sign == 1            