import cv2
import numpy as np
import imutils

#custom functions for alignment and pan-sharpening of images

def align_images(image, template, maxFeatures=500, keepPercent=0.2):
	"""
	image - rotated image
	template - original image
	maxFeatures - upper number of found keypoints
	keepPercent - percent of keypoints to keep, to eliminate noise
	"""

	# convert both the input image and template to grayscale
	imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # use ORB to detect keypoints and extract (binary) local
	# invariant features
	orb = cv2.ORB_create(maxFeatures)
	# extract keypoints and descriptors for image A
	(kpsA, descsA) = orb.detectAndCompute(imageGray, None)
	# extract keypoints and descriptors for image B
	(kpsB, descsB) = orb.detectAndCompute(templateGray, None)
	# match the features using Brute-Force matcher
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(descsA, descsB, None)

    # sort the matches by their distance (the smaller the distance,
	# the "more similar" the features are)
	matches = sorted(matches, key=lambda x:x.distance)
	# keep only the top matches
	keep = int(len(matches) * keepPercent)
	matches = matches[:keep]

	# visualize the matched keypoints
	matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
	matchedVis = imutils.resize(matchedVis, width=1000)
	cv2.imshow("Matched Keypoints", matchedVis)
	cv2.waitKey(0)

	#Alignment of images
    # allocate memory for the keypoints (x, y)-coordinates from the
	# top matches -- we'll use these coordinates to compute our homography matrix
	ptsA = np.zeros((len(matches), 2), dtype="float")
	ptsB = np.zeros((len(matches), 2), dtype="float")
	# loop over the top matches
	for (i, m) in enumerate(matches):
		# indicate that the two keypoints in the respective images map to each other
		ptsA[i] = kpsA[m.queryIdx].pt
		ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched points
	(H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
	# compute size (h, w) of the destination image
	(h, w) = template.shape[:2]
	#apply perspective transformation with matrix H to the first image
	aligned = cv2.warpPerspective(image, H, (w, h))
	# return the aligned image
	return aligned


def pansharpening(multi_image, pan_image, method='brovey', W = 0.4):
	"""
	pansharpening using one from three possible methods
	multi_image - multispectral image
	pan_image - panchromatic image
	method - method of pansharpening: brovey, simple_mean, esri
	W - weight to be used for Brovey algorithm
	"""
	#convert panochromic image to grayscale
	gray = cv2.cvtColor(pan_image, cv2.COLOR_BGR2GRAY)
	new_image = np.zeros(multi_image.shape, dtype='uint8')
	#channels
	B, G, R = 0, 1, 2

	if method == 'brovey':
		DNF = (gray) / (W * multi_image[:, :, B] + W * multi_image[:, :, G] + W * multi_image[:, :, R])

		for band in range(multi_image.shape[2]):
			 new_image[:, :, band] = multi_image[:, :, band]*DNF

	if method == 'simple_mean':
		for band in range(multi_image.shape[2]):
        		new_image[:, :, band] = 0.5 * (multi_image[:, :, band] + gray)

	if method == 'esri':
		ADJ = gray - multi_image.mean(axis = 2)
		for band in range(multi_image.shape[2]):
        		new_image[:, :, band] = multi_image[:, :, band] + ADJ

	return new_image