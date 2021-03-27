from rosws import *

def callback_image(raw):
    np_arr = np.fromstring(raw.data, np.uint8)
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image_np
cv2.namedWindow('image')

def nothing(x):
    pass
# create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0



if __name__ == '__main__':
    from sensor_msgs.msg import LaserScan
    
    rospy.init_node('Goodgame_websocket', anonymous=True)

    connect = WebsocketROSClient('127.0.0.1', 9090) # ip, port, name of client
    showMe = 1 # Show images
    peak_thresh = 10

    try:
        while not rospy.is_shutdown():
            
            # subscribe
            result =  connect.subscribe('g2_never_die/camera/rgb/compressed', CompressedImage())
            #callback_image(result)
            frame = (callback_image(result))
                # get current positions of all trackbars
            hMin = cv2.getTrackbarPos('HMin','image')
            sMin = cv2.getTrackbarPos('SMin','image')
            vMin = cv2.getTrackbarPos('VMin','image')

            hMax = cv2.getTrackbarPos('HMax','image')
            sMax = cv2.getTrackbarPos('SMax','image')
            vMax = cv2.getTrackbarPos('VMax','image')

            # Set minimum and max HSV values to display
            lower = np.array([hMin, sMin, vMin])
            upper = np.array([hMax, sMax, vMax])

            # Create HSV Image and threshold into a range.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            output = cv2.bitwise_and(frame,frame, mask= mask)

            # Print if there is a change in HSV value
            if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
                print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
                phMin = hMin
                psMin = sMin
                pvMin = vMin
                phMax = hMax
                psMax = sMax
                pvMax = vMax

            cv2.imshow("frame",output)
            cv2.waitKey(1)
            
    except KeyboardInterrupt:
        pass
    rospy.spin()
