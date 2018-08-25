import numpy as np
import cv2
import sys
import os
import matplotlib.pyplot as plt

############################
# Useful functions        
############################

def show_key_points(img, kp):
    displayed_img = cv2.drawKeypoints(img, kp, None)
    displayed_img = cv2.cvtColor(displayed_img, cv2.COLOR_BGR2RGB)
    plt.imshow(displayed_img), plt.show()
    
def show_matches(img1, kp1, img2, kp2, matches, matchesMask):
    draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 2)
    displayed_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    displayed_img = cv2.cvtColor(displayed_img, cv2.COLOR_BGR2RGB)
    plt.imshow(displayed_img), plt.show()

def show_key_points_two_image(img1, kp1, img2, kp2):
    displayed_img1 = cv2.drawKeypoints(img1, kp1, None)
    displayed_img2 = cv2.drawKeypoints(img2, kp2, None)
    displayed_img1 = cv2.cvtColor(displayed_img1, cv2.COLOR_BGR2RGB)
    displayed_img2 = cv2.cvtColor(displayed_img2, cv2.COLOR_BGR2RGB)
    plt.imshow(np.hstack([displayed_img1,displayed_img2])), plt.show()

def show_anaglyph(img1, img2):
    b1, g1, r1 = cv2.split(img1)
    b2, g2, r2 = cv2.split(img2)
    displayed_img = cv2.merge([b2,g2,g1])
    displayed_img = cv2.cvtColor(displayed_img,cv2.COLOR_BGR2RGB)
    plt.imshow(displayed_img, plt.show())

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape[:2]
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
    
#################
# Rectification #
#################

MIN_MATCH_COUNT = 10

def rectify(I1, I2, keep_which = 0):
    if I1.shape != I2.shape:
        raise ValueError ("left/right images must have the same dimensions")
    
    height, width = I1.shape[:2]

    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

    # Run a SURF detector on both images
    surf = cv2.xfeatures2d.SURF_create(400)

    kp1, des1 = surf.detectAndCompute(I1_gray, None)
    kp2, des2 = surf.detectAndCompute(I2_gray, None)

    # Display detected keypoints
    # show_key_points(I1,kp1)
    # show_key_points(I1,kp2)
    # show_key_points_two_image(I1,kp1,I2,kp2)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # Select good matches
    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    
    # Given sufficient matching
    if len(good) > MIN_MATCH_COUNT:

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)   
        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC)
        matchesMask = mask.ravel().tolist()

        # We select only inlier points
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]

        # Display good matches
        # show_matches(I1,kp1,I2,kp2,good,matchesMask)

        rv, homo1, homo2 = cv2.stereoRectifyUncalibrated (pts1.transpose().reshape ((-1)), 
                                                    pts2.transpose().reshape((-1)), F, (height, width))
        
        warp_mat_1 = np.mat(homo1)
        warp_mat_2 = np.mat(homo2)

        if keep_which == 0:
            I1_rectified = I1
            I2_rectified = cv2.warpPerspective(I2, np.linalg.inv(warp_mat_1) * warp_mat_2, (width, height))
        elif keep_which == 1:
            I1_rectified = cv2.warpPerspective(I1, np.linalg.inv(warp_mat_2) * warp_mat_1, (width, height))
            I2_rectified = I2
        else:
            I1_rectified = cv2.warpPerspective(I1, warp_mat_1, (width, height))
            I2_rectified = cv2.warpPerspective(I2, warp_mat_2, (width, height))

        # Display anaglyph images
        # show_anaglyph(I1_rectified,I2_rectified)

        # Display epipolar lines
        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        
        # lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
        # lines1 = lines1.reshape(-1,3)
        # I1_rectified_gray = cv2.cvtColor(I1_rectified, cv2.COLOR_BGR2GRAY)
        # I2_rectified_gray = cv2.cvtColor(I2_rectified, cv2.COLOR_BGR2GRAY)
        # display1, _ = drawlines(I1_rectified_gray,I2_rectified_gray,lines1,pts1[:10],pts2[:10])
        # # Find epilines corresponding to points in left image (first image) and
        # # drawing its lines on right image
        # lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
        # lines2 = lines2.reshape(-1,3)
        # display2,_ = drawlines(I2_rectified_gray,I1_rectified_gray,lines2,pts2[:10],pts1[:10])
        # plt.imshow(np.hstack([display1, display2])), plt.show()

        return I1_rectified, I2_rectified

    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
        return 

if __name__ == '__main__':
    # File names
    I1_name, I1_ext = os.path.splitext(sys.argv[1])
    I2_name, I2_ext = os.path.splitext(sys.argv[2])
    keep_which = int(sys.argv[3])

    # Read images and rectify them
    I1 = cv2.imread (sys.argv[1])
    I2 = cv2.imread (sys.argv[2])
    I1_rectified, I2_rectified = rectify (I1, I2, keep_which)
    cv2.imwrite (I1_name+"-rectified"+I1_ext, I1_rectified)
    cv2.imwrite (I2_name+"-rectified"+I2_ext, I2_rectified)
