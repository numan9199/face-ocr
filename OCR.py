import cv2 
import pytesseract
from preprocessing import Preprocessing
from PIL import Image
import io
from pytesseract import Output
from langdetect import detect_langs


# class Ocr():
#     @staticmethod
#     async def findtext():
#         return "text"

img = cv2.imread('thai id.jpeg')
custom_config = r'-l tha+eng --oem 3 --psm 11'

gray = Preprocessing.get_grayscale(img)
thresh = Preprocessing.thresholding(gray)
opening = Preprocessing.opening(gray)
canny = Preprocessing.canny(gray)


# m = pytesseract.image_to_string(thresh, config=custom_config)
# normalize_txt = ''
# for i in range(len(m)):
#     if ord(m[i]) != 32:
#         normalize_txt += m[i]
#     elif ord(m[i]) < 3585 :
#         normalize_txt += m[i]
# print(normalize_txt)
# with open('txt.txt','w',encoding="utf-8")as f :
#     for a in normalize_txt.splitlines():
#         f.write(a+'\n')
# #print(m)

# #draw box around a single character
# # h, w, c = img.shape
# # print(img.shape)
# # boxes = pytesseract.image_to_boxes(gray) 
# # for b in boxes.splitlines():
# #     b = b.split(' ')
# #     img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

# # cv2.imshow('img', img)
# # cv2.waitKey(0)

#draw box around text 
d = pytesseract.image_to_data(thresh, output_type=Output.DICT,config=custom_config)
n_boxes = len(d['text'])

for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)


