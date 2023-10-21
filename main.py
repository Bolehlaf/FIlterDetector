import cv2 as cv
import numpy as np


k_size = (35, 35)
filter_threshold = 100
hole_threshold = 130
hole_area_multiplication = 10


def find_filter_contours(image):
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    gauss = cv.blur(grayscale, k_size)

    ret, thresh = cv.threshold(gauss, filter_threshold, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    return contours


def find_large_hole(image):
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(grayscale, hole_threshold, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    areas = []

    for contour in contours:
        areas.append(cv.contourArea(contour))

    area_numpy = np.array(areas)
    median = np.median(area_numpy)

    i = 0
    for area in area_numpy:
        if area > (median * hole_area_multiplication):
            x, y, w, h = cv.boundingRect(contours[i])
            cv.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        i = i + 1


def draw_rectangles(image, contours):
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


if __name__ == '__main__':
    input_image = cv.imread('input/4.jpg')
    contours_output = find_filter_contours(input_image)

    find_large_hole(input_image)
    draw_rectangles(input_image, contours_output)

    cv.imwrite("output/output_4.png", input_image)
