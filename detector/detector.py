import cv2
import numpy as np
from collections import deque


class Detector():

    def __init__(self):
        self.buffer = deque([])

    def get_squares(self, image):
        (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(image, cv2.CV_32S)
        # Remove background stats
        stats = stats[1:]
        centroids = centroids[1:]
        # Area shouldn't be much different from median
        area = stats[:, cv2.CC_STAT_AREA]
        area_crit = abs(area - np.median(area)) < np.std(area)
        # Height and Width shouldn't be very different
        w = stats[:, cv2.CC_STAT_WIDTH]
        h = stats[:, cv2.CC_STAT_HEIGHT]
        idx = np.max((w/h, h/w), axis=0) < 1.2 * area_crit
        # Distance to main centroid should be small
        d = np.sqrt(np.sum(np.square(centroids - np.mean(centroids, axis=0)), axis=1))
        idx *= d < 4 * np.std(d)
        # Output coordinates
        x = stats[idx, cv2.CC_STAT_LEFT]
        y = stats[idx, cv2.CC_STAT_TOP]
        coordinates = np.array((x, y, x + w[idx], y + h[idx])).T
        return coordinates, centroids[idx]

    def detect_board(self, img):
        """ 
        Function to detect chess board 

        """
        # Convert to gray and threshold
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        norm_image = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        _, thresh_gray = cv2.threshold(norm_image.astype('uint8'), thresh=50, maxval=255, type=cv2.THRESH_BINARY_INV)
        
        ## FIND BOARD
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
        closing = cv2.morphologyEx(thresh_gray, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

        ## FIND SQUARES
        # Morph opening to filter
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
        opening = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN, kernel)
        # Find black squares
        coor, centroids = self.get_squares(opening)
        # Add markings
        for c, (x, y, w, h) in zip(centroids, coor):
            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 3)
            cv2.circle(img, (int(c[0]), int(c[1])), 4, (0, 0, 255), -1)

        return img

    def show_stream(self, buffer, detect=False):
        """ 
        Show stream of Video with detected chess board 

        """
        while True:
            if len(buffer.image_buffer) <= 0:
                continue
            # Read image
            image = buffer.image_buffer.pop()
            im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # Detect chess board
            if detect:
                im = self.detect_board(im)
            # Show
            cv2.imshow('Video', im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def save_stream(self, buffer):
        img_array = []
        while True:
            if len(buffer.image_buffer) > 0:
                image = buffer.image_buffer.pop()
            else:
                continue

            im = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            cv2.imshow('Video', im)
            img_array.append(im)

            height, width, layers = im.shape
            size = (width, height)

        cv2.destroyAllWindows()

        out = cv2.VideoWriter(
            'vid.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5.0, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
