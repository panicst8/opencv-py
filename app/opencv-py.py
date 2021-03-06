""" opencv thingy """

import sys
import os
import cv2
import numpy as np
from typing import Any

PACKAGE_PARENT = ".."
SCRIPT_DIR = os.path.dirname(
    os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__)))
)
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from app.utils import stackImages  # noqa: E402


def face_detection() -> None:
    """ detect faces from image """

    # Load target image
    img = cv2.imread("assets/lena.png")
    # there are many prebuilt harrcascade types for identifying other things
    faceCascade = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")
    # Convert to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # where the magic happens
    faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

    # draw bounding box around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show results
    cv2.imshow("Lena", img)
    cv2.waitKey(0)


def getContours(img: Any, imgCanny: Any) -> Any:
    """ get contours """

    _img = img.copy()
    # cv2.RETR_EXTERNAL - retreive outer externl contours
    contours, hiarchy = cv2.findContours(
        imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    t, r, s, c = (0, 0, 0, 0)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(_img, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            typeOfShape = "?"
            if objCor == 3:
                t += 1
                typeOfShape = f"{t} Tri"
            if objCor == 4:
                aspectRatio = w / float(h)
                if aspectRatio > 0.98 and aspectRatio < 1.02:
                    s += 1
                    typeOfShape = f"{s} Sqr"
                else:
                    r += 1
                    typeOfShape = f"{r} Rec"
            if objCor > 4:
                c += 1
                typeOfShape = f"{c} Cir"
            cv2.putText(
                _img,
                typeOfShape,
                (x + (w // 2) - 10, y + (h // 2) - 5),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
    print(f"Found: \n{t} Triangles\n{s} Squares")
    print(f"{r} Rectangles\n{c} Circles")

    return _img


def shape_detection() -> None:
    """ Tut 7 shape detct """
    img = cv2.imread("assets/shapes.png")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, 50, 50)
    imgBlank = np.zeros_like(img)
    imgContour = getContours(img, imgCanny)

    imgStack = stackImages(
        0.8, ([img, imgGray, imgBlur], [imgCanny, imgBlank, imgContour])
    )
    cv2.imshow("Shapes", imgStack)
    cv2.waitKey(0)
    # kernal = np.ones((5, 5), np.uint8)
    # cv2.imshow("Shapes", img)
    # cv2.waitKey(0)
    # cv2.imshow("Shapes Gray", imgGray)
    # cv2.waitKey(0)
    # cv2.imshow("Shapes Blur", imgBlur)
    # cv2.waitKey(0)


def empty(val: Any) -> None:
    """ placeholder """


def color_detection() -> None:
    """ detect colors """

    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 640, 240)
    cv2.createTrackbar("Hue Min", "Trackbars", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "Trackbars", 19, 179, empty)
    cv2.createTrackbar("Sat Min", "Trackbars", 110, 255, empty)
    cv2.createTrackbar("Sat Max", "Trackbars", 240, 255, empty)
    cv2.createTrackbar("Val Min", "Trackbars", 153, 255, empty)
    cv2.createTrackbar("Val Max", "Trackbars", 255, 255, empty)

    while True:
        img = cv2.imread("assets/lambo.png")
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
        h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
        s_min = cv2.getTrackbarPos("Sat Min", "Trackbars")
        s_max = cv2.getTrackbarPos("Sat Max", "Trackbars")
        v_min = cv2.getTrackbarPos("Val Min", "Trackbars")
        v_max = cv2.getTrackbarPos("Val Max", "Trackbars")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv2.inRange(imgHSV, lower, upper)

        imgResult = cv2.bitwise_and(img, img, mask=mask)

        # cv2.imshow("Masked Image", imgResult)
        # cv2.imshow("Lambo", img)
        # cv2.imshow("HSV Lambo", imgHSV)
        # cv2.imshow("Lambo Mask", mask)

        imgStack = stackImages(0.6, ([img, imgHSV], [mask, imgResult]))
        cv2.imshow("Stacked Images", imgStack)

        cv2.waitKey(1)


def img_stacking() -> None:
    """ use cv2 image stacking functions """
    img = cv2.imread("assets/finn.jpg")

    imgHor = np.hstack((img, img))
    imgVir = np.vstack((img, img))

    cv2.imshow("Horizonal img stack", imgHor)
    cv2.waitKey(0)
    cv2.imshow("Virtical img stack", imgVir)
    cv2.waitKey(0)


def skew() -> None:
    """ skew playing card """

    width, height = 250, 350

    pts_1 = np.float32([[111, 219], [287, 188], [154, 482], [352, 440]])
    pts_2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    img = cv2.imread("assets/cards.jpg")

    matrix = cv2.getPerspectiveTransform(pts_1, pts_2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))

    cv2.imshow("Cards before skew", img)
    cv2.waitKey(0)

    cv2.imshow("Card skewed", imgOutput)
    cv2.waitKey(0)


def diy_images() -> None:
    """ make images from scratch """
    img2 = np.zeros((512, 512, 3), np.uint8)
    # print(img2)

    # img2[:] = 255, 0, 0 # make image blue
    cv2.line(img2, (10, 10), (100, 100), (0, 255, 0), 5)
    cv2.rectangle(img2, (10, 10), (100, 100), (0, 255, 0), 5)
    cv2.circle(img2, (150, 150), 100, (0, 0, 255), cv2.FILLED)
    cv2.putText(
        img2, "Hello World", (200, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2
    )

    cv2.imshow("Black Image", img2)
    cv2.waitKey(0)


def misc_examples() -> None:
    """ smaller examples grouped into func """
    img = cv2.imread("assets/finn.jpg")
    cv2.imshow("output", img)
    cv2.waitKey(0)

    # image change to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray output", imgGray)
    cv2.waitKey(0)

    # image blur
    imgBlur = cv2.GaussianBlur(img, (7, 7), 0)
    cv2.imshow("blur output", imgBlur)
    cv2.waitKey(0)

    # resizing image
    imgResize = cv2.resize(img, (300, 200))
    print("img: ", img.shape, "\nimResize: ", imgResize.shape)
    cv2.imshow("resized output", imgResize)
    cv2.waitKey(0)

    # crop image
    imgCropped = img[0:500, 200:500]
    cv2.imshow("cropped output", imgCropped)
    cv2.waitKey(0)

    imgCanny = cv2.Canny(img, 100, 100)
    cv2.imshow("Canny output", imgCanny)
    cv2.waitKey(0)

    kernal = np.ones((5, 5), np.uint8)

    imgDilation = cv2.dilate(imgCanny, kernal, iterations=2)
    cv2.imshow("dilation output", imgDilation)
    cv2.waitKey(0)

    imgEroded = cv2.erode(imgDilation, kernal, iterations=2)
    cv2.imshow("eroded output", imgEroded)
    cv2.waitKey(0)


def video_examples() -> None:
    """ load video and use webcam """

    cap = cv2.VideoCapture("assets/finn.mp4")

    while True:
        success, img = cap.read()
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap = cv2.VideoCapture(0)

    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(10, 100)  # brightness setting

    while True:
        success, img = cap.read()
        cv2.imshow("Video", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


def main() -> None:
    """ main """

    face_detection()
    quit()


if __name__ == "__main__":
    main()
