import cv2
import numpy as np
import time
import RPi.GPIO as GPIO

# GPIO Init
GPIO.setmode(GPIO.BCM)

# Stereo camera
stereo = cv2.StereoSGBM_create(minDisparity = 0,
                                numDisparities = 96,
                                blockSize = 5,
                                P1 = 3 * 4 * 5 ** 2,
                                P2 = 3 * 32 * 5 ** 2,
                                disp12MaxDiff = 1,
                                preFilterCap = 63,
                                uniquenessRatio = 10,
                                speckleWindowSize = 100,
                                speckleRange = 32,
                                mode = 2
                                )
leftC = cv2.VideoCapture(3, cv2.CAP_V4L2)
rightC = cv2.VideoCapture(0, cv2.CAP_V4L2)
leftC.set(3, 400)
leftC.set(4, 400)
rightC.set(3, 400)
rightC.set(4, 400)

def detect(srcImg):
    hsvImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2HSV)
    maskRed1 = cv2.inRange(hsvImg, (0, 100, 20), (10, 255, 255))
    maskRed2 = cv2.inRange(hsvImg, (160, 100, 20), (179, 255, 255))
    maskRed = maskRed1 + maskRed2
    cv2.imshow("red", maskRed)
    maskRed = cv2.erode(maskRed, np.ones((5, 5), np.uint8), iterations=1)
    maskRed = cv2.dilate(maskRed, np.ones((3, 3), np.uint8), iterations=1)
    contours, hierarchy = cv2.findContours(maskRed, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
    start_row = end_row = start_col = end_col = None
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 25 and abs(w - h) < 5:
            start_row = y
            end_row = y + h
            start_col = x
            end_col = x + w
            break

    return start_row, end_row, start_col, end_col

def main():
    # GPIO Pin
    motorL = 19
    motorR = 12
    GPIO.setup(motorL,GPIO.OUT)
    GPIO.setup(motorR,GPIO.OUT)

    GPIO.setwarnings(False)

    pwmL = GPIO.PWM(motorL, 1000)
    pwmR = GPIO.PWM(motorR, 1000)

    pwmL.start(0)
    pwmR.start(0)
    
    # PID controller
    eprev = eintegral = 0.
    target = 40. #cm (minDistance =35cm)
    kp, kd, ki = 1., 0.025, 0.
    
    prevT = time.time()
    while True:
        ret, leftImg = leftC.read()
        ret, rightImg = rightC.read()
        
        start_row, end_row, start_col, end_col = detect(leftImg)
        if(start_row == None): continue

        leftImg = cv2.cvtColor(leftImg, cv2.COLOR_BGR2GRAY)
        rightImg = cv2.cvtColor(rightImg, cv2.COLOR_BGR2GRAY)

        # Disparity map
        disp = stereo.compute(leftImg, rightImg).astype('float32') / 16.0 / 96

        dispCrop = disp[start_row:end_row, start_col:end_col]
        dispCrop[dispCrop < 0] = 0
        
        #Distance object
        pos = 36.50376 / np.sum(dispCrop) * np.count_nonzero(dispCrop) - 5.81458
        
        #calculate signal u -- PID controller
        # time difference
        deltaT = time.time() - prevT
        prevT = time.time()
        
        # error
        e = pos - target

        # derivative
        dedt = (e-eprev)/deltaT

        # integral
        eintegral = eintegral + e*deltaT

        # control signal
        u = kp*e + kd*dedt + ki*eintegral
        
        if u > 100:
            duty = 100
        elif 50 < u < 100:
            duty = int(u)
        elif 0 < u < 50:
            duty = 50
        else:
            duty = 0
            
        # set duty cycle
        pwmL.ChangeDutyCycle(duty)
        pwmR.ChangeDutyCycle(duty)

        eprev = e
        
        cv2.waitKey(1)
    leftC.release()
    rightC.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
