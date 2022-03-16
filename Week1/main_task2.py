import utils as ut
import voc_evaluation
import matplotlib.pyplot as plt
import numpy as np

# PATHS
annotations_path = 'data/ai_challenge_s03_c010-full_annotation.xml'
video_path = 'data/AICity_data/train/S03/c010/vdo.avi'
detections_path = {
    'rcnn': 'data/AICity_data/train/S03/c010/det/det_mask_rcnn.txt',
    'ssd': 'data/AICity_data/train/S03/c010/det/det_ssd512.txt',
    'yolo': 'data/AICity_data/train/S03/c010/det/det_yolo3.txt'
}

if __name__ == '__main__':

    display = False
    test_det = False
    generate_noise = True
    noisy_percent = 0.5
    compare_detections_plot = False

    annotations = ut.read_annotations(annotations_path)

    if test_det:
        detections = ut.annotations_to_detections(
            annotations, generate_noise, noisy_percent)
    else:
        detections = ut.read_detections(detections_path['yolo'])
    if compare_detections_plot:
        ut.plot_multiple_IoU(detections_path,annotations)
    score_for_each_frame = []
    
    for frame_id in range(len(annotations)):
        score_for_each_frame.append(ut.get_frame_iou(
            annotations[frame_id], detections[frame_id]))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(0, len(annotations), 1), score_for_each_frame)
    ax.set(xlabel='frame', ylabel='IoU')
    ax.set_ylim(0, 1)
    ax.set_xlim(0,len(annotations))
    plt.savefig('task2_plot.png')

    #rec, prec, ap = voc_evaluation.voc_eval(annotations, detections)

    # Overall detections
    rec, prec, ap = voc_evaluation.voc_eval(annotations, detections)
    print(rec)
    print(prec)
    print('AP score equal to {}'.format(ap))
    print('mean IoU {}'.format(np.mean(score_for_each_frame)))
