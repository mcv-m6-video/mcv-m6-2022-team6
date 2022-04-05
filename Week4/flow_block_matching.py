import numpy as np
import cv2

METRICS = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
		   'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']


def exhaustive_search_cv2(template: np.ndarray, target: np.ndarray, metric='cv2.TM_CCOEFF_NORMED'):
	"""
	search at all possible positions in target
	"""
	# evaluate the openCV metric
	metric = eval(metric)
	result = cv2.matchTemplate(target, template, metric)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
	cv2.imshow("target",target)
	cv2.imshow("sqare",template)
	cv2.imshow("result",result)

	if metric in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
		pos = min_loc
	else:
		pos = max_loc
	return pos


def exhaustive_search_own(template: np.ndarray, target: np.ndarray, block_size, metric="SSD"):
	"""
	search at all possible positions in target
	"""
	h, w = target.shape[:2]

	err_fn = {
		"SAD": lambda i, j: np.sum(np.abs(template - target[i:i + block_size, j:j + block_size])),
		"SSD": lambda i, j: np.sum(np.square(template - target[i:i + block_size, j:j + block_size])),
		"MAE": lambda i, j: np.sum(np.abs(template - target[i:i + block_size, j:j + block_size])) / block_size ** 2,
		"MSE": lambda i, j: np.sum(np.square(template - target[i:i + block_size, j:j + block_size])) / block_size ** 2
	}

	errors = np.zeros((h - block_size, w - block_size));
	if err_fn[metric] is None:
		return 0, 0

	for i in range(0, h - block_size):
		for j in range(0, w - block_size):
			errors[i, j] = err_fn[metric](i, j);

	return np.unravel_index(errors.argmin(), errors.shape);


def logarithmic_search(template: np.ndarray, target: np.ndarray, step=4):
	orig = ((target.shape[0] - template.shape[0]) // 2, (target.shape[1] - template.shape[1]) // 2)
	step = (min(step, orig[0]), min(step, orig[1]))
	while step[0] > 1 and step[1] > 1:
		min_dist = np.inf
		pos = orig
		for i in [orig[0] - step[0], orig[0], orig[0] + step[0]]:
			for j in [orig[1] - step[1], orig[1], orig[1] + step[1]]:

				def _distance(x1, x2):
					return np.mean((x1 - x2) ** 2)

				dist = _distance(template, target[i:i + template.shape[0], j:j + template.shape[1]])
				if dist < min_dist:
					pos = (i, j)
					min_dist = dist
		orig = pos
		step = (step[0] // 2, step[1] // 2)
	return orig


def get_optical_flow(img1: np.ndarray, img2: np.ndarray, block_size=32, area=32, mode="forward",
					 method="log", show_preview=False, log_step = 4):

	if mode == "backward":
		img1, img2 = img2, img1

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

			if method == "cv2":
				displacement  = exhaustive_search_cv2(patch, search_patch)
			elif method == "log":
				displacement = logarithmic_search(patch, search_patch, step=log_step)
			elif method == "full":
				displacement = exhaustive_search_own(patch, search_patch, block_size)

			v_flow = int(displacement[1]) - (top_l[1]- top_l_search[1])
			u_flow = int(displacement[0]) - (top_l[0]- top_l_search[0])
			flow[ih:ih + block_size, iw:iw + block_size] = [u_flow, v_flow]

			if show_preview:
				preview = img1.copy()
				preview = cv2.rectangle(preview, top_l_search, bot_r_search, (255, 0, 0));
				preview = cv2.rectangle(preview, top_l, bot_r, (0, 0, 255));
				preview = cv2.rectangle(preview, (u_flow + top_l[0], v_flow + top_l[1]),
										(u_flow + top_l[0] + block_size, v_flow + top_l[1] + block_size), (0, 255, 0));
				#preview = cv2.drawMarker(preview, (u_flow + top_l[0], v_flow + top_l[1]), (0, 255, 0))
				cv2.imshow("Preview", preview)
				preview2 = img2.copy()
				preview2 = cv2.rectangle(preview2, (u_flow + top_l[0], v_flow + top_l[1]),
										(u_flow + top_l[0] + block_size, v_flow + top_l[1] + block_size), (0, 255, 0));

				cv2.imshow("Preview2",preview2)
				#cv2.waitKey(0) 

	flow = np.dstack((flow[:, :, 0], flow[:, :, 1], np.ones_like(flow[:, :, 0])))
	return flow
