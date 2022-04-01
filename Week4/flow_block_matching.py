import numpy as np
import cv2

METRICS = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
		   'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


def exhaustive_search(template: np.ndarray, target: np.ndarray, metric='cv2.TM_CCORR_NORMED'):
	"""
	search at all possible positions in target
	"""
	# evaluate the openCV metric
	metric = eval(metric)
	result = cv2.matchTemplate(template, target, metric)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

	if metric in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
		pos = min_loc
	else:
		pos = max_loc
	return pos


def get_optical_flow(img1: np.ndarray, img2: np.ndarray, block_size=16, area=16):

	h, w = img1.shape[:2]
	flow = np.zeros((h, w, 2), dtype=float)

	for ih in range(0, h - block_size, block_size):
		for iw in range(0, w - block_size, block_size):
			top_l = (iw, ih);
			top_l_search = (max(iw - area, 0), max(ih - area, 0));
			bot_r = (iw + block_size, ih + block_size);
			bot_r_search = (min(bot_r[0] + area, w), min(bot_r[1] + area, h));

			patch = img1[top_l[1]:bot_r[1], top_l[0]:bot_r[0]];
			search_patch = img2[top_l_search[1]:bot_r_search[1], top_l_search[0]:bot_r_search[0]]

			displacement = exhaustive_search(patch, search_patch)

			v_flow = displacement[1] - (ih - top_l[1])
			u_flow = displacement[0] - (iw - top_l[0])
			flow[ih:ih + block_size, iw:iw + block_size] = [u_flow, v_flow]


			if False:
				preview = img1.copy()
				preview = cv2.rectangle(preview, top_l_search, bot_r_search, (255, 0, 0));
				preview = cv2.rectangle(preview, top_l, bot_r, (0, 0, 255));
				preview = cv2.drawMarker(preview, (u_flow + top_l[0], v_flow + top_l[1]), (0, 255, 0))
				cv2.imshow("Preview", preview)
				cv2.waitKey(10);

	flow = np.dstack((flow[:, :, 0], flow[:, :, 1], np.ones_like(flow[:, :, 0])))
	return flow
