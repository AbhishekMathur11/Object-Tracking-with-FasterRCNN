def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.

    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

    Returns:
        float: intersection-over-onion of bbox1, bbox2
    """

    # TODO: Calculate the coordinates for the intersection rectangle


    # TODO: Calculate the coordinates for the intersection rectangle
    # print(f"Calculating IoU for bbox1: {bbox1}, bbox2: {bbox2}")

    intersect_height = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    intersect_width = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])

    if intersect_height <= 0 or intersect_width <= 0:
        return 0

    size_intersection = intersect_height * intersect_width
    box1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    box2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    size_union = box1_area + box2_area - size_intersection

    iou_value = size_intersection / size_union
        # print(f"IoU calculated: {iou_value}")
    return iou_value
def track_iou(detections, sigma_l, sigma_h, sigma_iou, t_min):
    """
    Implements a simple IoU-based multi-object tracker. Matches detections to existing tracks based on IoU.
    Detections with IoU above a threshold are linked to existing tracks; otherwise, new tracks are created.

    See "High-Speed Tracking-by-Detection Without Using Image Information by Bochinski et al. for
    more information.

    Args:
         detections (list): list of detections per frame
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.

    Returns:
        list: list of tracks. Each track is a dict containing 'bboxes': a list of bounding boxes, 'max_score': the
        maximum detection score, and 'start_frame': the frame index of the first detection.
    """

    # Initialize an empty list to store active and completed tracks
    tracks_active = []
    tracks_finished = []

    # Loop over each frame’s detections
    for frame_num, detections_frame in enumerate(detections):
        # TODO: Apply low threshold sigma_l to filter low-confidence detections
        dets = []
        for det in detections_frame:
            if det['score'] >= sigma_l:
                dets.append(det)

        updated_tracks = []
        for track in tracks_active:
            track_updated = False

            # If there are detections for this frame
            if len(dets) > 0:
                # TODO: get det with highest iou
                iou_valid = None
                best_match = {'iou': 0, 'bbox': None}
                for det in dets:
                    iou_track = iou(track['bboxes'][-1], det['bbox'])
                    if iou_track > best_match['iou']:
                        best_match['iou'] = iou_track
                        best_match['bbox'] = det['bbox']
                        iou_valid = det

                # TODO: If IoU of best_match, exceeds sigma_iou, then extend the track by adding the detection to the track,
                # update the max_score, then remove that detection from the dets. Remember to set track_updated to True.
                # If IoU of best_match, exceeds sigma_iou, then extend the track by adding the detection to the track,
                # update the max_score, then remove that detection from the dets. Remember to set track_updated to True.
                if best_match['iou'] >= sigma_iou:
                    track['bboxes'].append(list(map(int,best_match['bbox'])))
                    track['max_score'] = max(track['max_score'], det['score'])
                    dets.remove(iou_valid)
                    track_updated = True
                    updated_tracks.append(track)

            # If track was not updated
            if not track_updated:
                # TODO: finish track when the conditions are met by appending the track to tracks_finished
                if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min:
                    tracks_finished.append(track)
                    tracks_active.remove(track)

        # create new tracks

        new_tracks = [{'bboxes': [list(map(int,det['bbox']))], 'max_score': det['score'], 'start_frame': frame_num} for det in dets]
        tracks_active = updated_tracks + new_tracks

    # finish all remaining active tracks
    tracks_finished += [track for track in tracks_active
                        if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]

    return tracks_finished

def run_tracker(frames):
    # Track objects in the video
    detections = []
    for frame_num, (image, bboxes, confidences, class_ids) in enumerate(frames):
        dets = []
        for bbox, confidence, class_id in zip(bboxes, confidences, class_ids):
            dets.append({'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],
                         'score': confidence,
                         'class': class_id})
        detections.append(dets)

    print('Running tracker...')
    tracks = track_iou(detections, sigma_l=0.4, sigma_h=0.7, sigma_iou=0.3, t_min=2)
    print('Tracker finished!')
    return tracks
