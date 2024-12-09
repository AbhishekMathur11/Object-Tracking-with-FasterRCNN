def get_model(device):
    """
    Load the pretrained model + inference transform
    """
    # Load the model
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    # Load the model onto the computation device
    model = model.eval().to(device)
    # inference transform
    preprocess = weights.transforms()

    return model, preprocess


def predict(image, model, device, detection_threshold):
    """
    Predicts bounding boxes, scores, and class labels for objects detected in an image.
    Only returns detections with confidence above the specified threshold.

    Args:
        image (torch.Tensor): The input image tensor.
        model (torchvision.models.detection.FasterRCNN): The object detection model.
        device (torch.device): The device to perform computations on.
        detection_threshold (float): Confidence threshold for filtering detections.

    Returns:
        boxes (numpy.ndarray): Bounding boxes of detected objects above the confidence threshold. Shape (N, 4),
            where N is the number of detections. Bbox format: (x1, y1, x2, y2)
        scores (numpy.ndarray): Confidence scores for the detected objects. Shape (N,)
        labels (numpy.ndarray): Class labels for the detected objects. Shape (N,)
    """
    # raise NotImplementedError('not implemented')

    # TODO: Move the input image to the specified device (GPU)


    image = image.to(device)

    # TODO: Add a batch dimension to the image tensor
    image = image.unsqueeze(0)

    # TODO: Run the forward pass (with torch.no_grad()) to get model outputs
    with torch.no_grad():
        prediction = model(image)

    # TODO: Extract the scores, bounding boxes, and labels from the model outputs
    scores = prediction[0]['scores'].cpu().numpy()
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    # TODO: Apply the detection threshold to filter out low-confidence predictions
    keeper = scores >= detection_threshold
    scores = scores[keeper]
    boxes = boxes[keeper]
    labels = labels[keeper]
    return boxes, scores, labels


def run_detector(image_path, model, preprocess, device, det_threshold=0.9):
    """
    Runs the object detector on a given image and retrieves bounding boxes, confidence scores,
    and class labels for detected objects.

    Args:
        image_path (str): Path to the image file to detect objects in.
        model (torchvision.models.detection.FasterRCNN): The object detection model.
        preprocess (callable): Preprocessing function for the image.
        det_threshold (float): Confidence threshold for detections.

    Returns:
        image_np (numpy.ndarray): Original image in numpy array format (for visualization later)
        bboxes (numpy.ndarray): Bounding boxes of detected objects.
        confidences (numpy.ndarray): Confidence scores for the detected objects.
        class_ids (numpy.ndarray): Class labels for the detected objects.
    """
    # Read image to tensor (0-255 uint8)
    image_torch = read_image(image_path)
    image_torch = image_torch[:3,:,:]

    #image = Image.open(image_path).convert("RGB") # Ensure the image is loaded as RGB
    image_tensor = preprocess(image_torch)
    print(image_tensor.shape)
    image_np = image_torch.permute(1,2,0).numpy()



    # TODO: Apply the preprocess to preprocess the image (normalization, etc.) (see more at https://pytorch.org/vision/0.20/transforms.html)


    # TODO: Run the predict function on image_processed to obtain bounding boxes, scores, and class IDs
    bboxes, confidences, class_ids = predict(image_tensor, model, device = device, detection_threshold = det_threshold)
    #image_np = image_tensor[0].cpu().numpy() # Get the image from the list


    return (image_np, bboxes, confidences, class_ids)
