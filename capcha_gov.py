import cv2
import numpy as np
import imutils
import pytesseract
import pyautogui
import time
# print(pyautogui.position())

# to generate new capcha
# time.sleep(5)
# W = 1173-948
# H = 419-374
# pyautogui.screenshot(r'C:\Users\akash148363\PycharmProjects\Personal\captha4.tiff',region=(948,374,W,H))
# exit()
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\akash148363\Downloads\tesseract.exe'

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

img = cv2.imread(r"C:\Users\akash148363\PycharmProjects\Personal\captha4.tiff")
blur = cv2.medianBlur(img,1)
gray = cv2.cvtColor(blur ,cv2.COLOR_BGR2GRAY)

_, thres = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
_, thres2 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
cv2.imshow('thres',thres)
cv2.waitKey(0)
cv2.destroyAllWindows()
cnts, h = cv2.findContours(thres,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts, boundingBoxes = sort_contours(cnts)
print(f'No. of cnts {len(cnts)} found')
im2 = img.copy()
for cnt in cnts:
    x, y, w, h = cv2.boundingRect(cnt)

    # Drawing a rectangle on copied image
    rect = cv2.rectangle(im2, (x-5, y-5), (x + w+5, y + h+5), (0, 255, 0), 1)

    # Cropping the text block for giving input to OCR
    # cropped = im2[y-5:y + h+5, x-5:x + w+5]
    cropped = thres2[y - 5:y + h + 5, x - 5:x + w + 5]
    # percent by which the image is resized
    src = cropped
    scale_percent = 120

    # calculate the 50 percent of original dimensions
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)

    # dsize
    dsize = (width, height)

    # resize image
    cropped = cv2.resize(src, dsize)
    cv2.imshow('cropped ', cropped )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    text = pytesseract.image_to_string(cropped,lang='eng',
                                       config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    print(text)
    # cv2.imshow('image',cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


cv2.imshow('image',im2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.boundingRect()

# def draw_contour(image, c, i):
#
#     # compute the center of the contour area and draw a circle
#     # representing the center
#     M = cv2.moments(c)
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#
#     # draw the countour number on the image
#     cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
#         1.0, (255, 255, 255), 2)
#
#     # return the image with the contour number drawn on it
#     return image
#
# # construct the argument parser and parse the arguments
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-i", "--image", required=True, help="Path to the input image")
# # ap.add_argument("-m", "--method", required=True, help="Sorting method")
# # args = vars(ap.parse_args())
# img_path = r'C:\Users\akash148363\PycharmProjects\Personal\margin.png'
# # load the image and initialize the accumulated edge image
# image = cv2.imread(img_path)
# accumEdged = np.zeros(image.shape[:2], dtype="uint8")
#
# # loop over the blue, green, and red channels, respectively
# for chan in cv2.split(image):
#     # blur the channel, extract edges from it, and accumulate the set
#     # of edges for the image
#     chan = cv2.medianBlur(chan, 11)
#     edged = cv2.Canny(chan, 50, 200)
#     accumEdged = cv2.bitwise_or(accumEdged, edged)
#
# # show the accumulated edge map
# cv2.imshow("Edge Map", accumEdged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # find contours in the accumulated image, keeping only the largest
# # ones
# cnts = cv2.findContours(accumEdged.copy(), cv2.RETR_EXTERNAL,
#     cv2.CHAIN_APPROX_SIMPLE)
# print(f'No of cnts found {len(cnts)}')
# cnts = imutils.grab_contours(cnts)
# cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
# orig = image.copy()
#
# # loop over the (unsorted) contours and draw them
# for (i, c) in enumerate(cnts):
#     orig = draw_contour(orig, c, i)
#
# # show the original, unsorted contour image
# cv2.imshow("Unsorted", orig)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # sort the contours according to the provided method
# (cnts, boundingBoxes) = sort_contours(cnts, method='left-to-right')
#
# # loop over the (now sorted) contours and draw them
# for (i, c) in enumerate(cnts):
#     draw_contour(image, c, i)
#
# # show the output image
# cv2.imshow("Sorted", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
