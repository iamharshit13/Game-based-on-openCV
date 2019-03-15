import cv2
import numpy as np

def main():
    
    w = 800
    h = 600    
    
    cap = cv2.VideoCapture(0)
    
    cap.set(3, w)
    cap.set(4, h)
    

    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()


    while ret:
        
        d = cv2.absdiff(frame1, frame2)
        
        g = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
        
        b = cv2.GaussianBlur(g, (5, 5), 1)
        
        ret, th = cv2.threshold( b, 20, 255, cv2.THRESH_BINARY)
    
        d = cv2.dilate(th, np.ones((1, 1), np.uint8), iterations=2 )
        
        e = cv2.erode(d, np.ones((3, 3), np.uint8), iterations=2 )
        
        img, c, h = cv2.findContours(e, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(frame1, c, -1, (100, 200, 100), 2)

        cv2.imshow("Original", frame2)
        cv2.imshow("Output", frame1)
        if cv2.waitKey(1) == 27: # ESC to exit
            break
        
        frame1 = frame2
        ret, frame2 = cap.read()

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()