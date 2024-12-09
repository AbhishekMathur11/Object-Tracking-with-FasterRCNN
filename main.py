import utils
import model
import multi_obj


if not os.path.exists('car_frames_simple.zip'):
  !wget https://www.andrew.cmu.edu/user/kvuong/car_frames_simple.zip -O car_frames_simple.zip
  !unzip -qq "car_frames_simple.zip"
  print("downloaded and unzipped data")


# Define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cpu':
  print('!!! WARNING: USING CPU ONLY, THIS WILL BE VERY SLOW !!!')

# First, load the model and preprocessor
model, preprocess = get_model(device)

# TODO: either use wget or manually upload the image to temporary storage (please don't use the same image as the example in the pdf)
# !wget "https://media.formula1.com/image/upload/t_16by9South/f_auto/q_auto/v1711253924/trackside-images/2024/F1_Grand_Prix_of_Australia/2108303848.jpg" -O example.png
!wget "https://assets.simpleviewinc.com/simpleview/image/upload/c_limit,h_1200,q_75,w_1200/v1/clients/texas/discoverdenton_Instagram_3163_ig_18149046172184896_938dfe6c-f02c-4626-ac96-ba2b906eed74.jpg" -O example.png
image_path = "/content/example.png"

# Run the detector on the image
output_det = run_detector(image_path, model, preprocess, device, det_threshold=0.7)  # Lowered threshold
image, bboxes, confidences, class_ids = output_det

# Draw the boxes and display the image
image_with_boxes = draw_boxes(bboxes, class_ids, image)
plt.imshow(image_with_boxes)
plt.axis('off')
plt.tight_layout()
plt.show()
# TODO: run object detector on every image inside the data folder
image_folder = "./car_frames_simple"
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

output_detections = []
for image_path in image_paths:
    output_det = run_detector(image_path, model, preprocess, device, det_threshold=0.9)
    output_detections.append(output_det)

# Visualize a few images (first and last image for example)
indices = [0, len(output_detections) - 1]
for idx in indices:
    image, bboxes, confidences, class_ids = output_detections[idx]
    # Draw the boxes and display the image
    image_with_boxes = draw_boxes(bboxes, class_ids, image)
    plt.imshow(image_with_boxes)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# TODO: From the detections, run the tracker to obtain a list of tracks
output_tracks = run_tracker(output_detections)

image_vis_list = draw_multi_tracks(output_detections, output_tracks)

print(image_vis_list[0].shape)
# TODO: Visualize a few images (here we show first, middle, and last image for example)
indices = [0, len(output_detections) // 2, len(output_detections) - 1]
#indices = np.arange(len(output_detections))
for idx in indices:
    plt.imshow(image_vis_list[idx])
    plt.axis('off')
    plt.tight_layout()

    plt.show()

