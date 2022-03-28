import detectron2
import pickle
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import numpy as np

CLASS_CAR = 2;
TOTAL_FRAMES = 2141
SHOW_VIDEO = False

def configure_detectron():
	cfg = get_cfg()
	cfg.MODEL.DEVICE = 'cpu'
	# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
	cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
	# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
	cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
	return cfg


def filter_class(outputs, class_type):
	pred_classes = outputs["instances"].pred_classes.numpy();
	is_car = pred_classes == class_type
	instances = outputs['instances'][is_car];
	return instances


if __name__ == '__main__':

	cfg = configure_detectron()
	predictor = DefaultPredictor(cfg)

	for i in range(1, TOTAL_FRAMES + 1):
		print("Frame %04d of %04d" % (i, TOTAL_FRAMES));
		im = cv2.imread("../data/images/%04d.jpeg" % i)
		outputs = predictor(im)
		instances = filter_class(outputs, CLASS_CAR)
		#np.save('../data/detections/frame_bboxes_%04d.npy' % i, instances.pred_boxes.tensor.numpy())

		v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
		out = v.draw_instance_predictions(instances)

		# Show
		if SHOW_VIDEO:
			cv2.imshow('Video', out.get_image()[:, :, ::-1])
			if cv2.waitKey(10) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break
