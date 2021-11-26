""" This demo implements the affordance detection pipeline described in Apicella, T. et al., "An Affordance Detection Pipeline for Resource-Constrained Devices", ICECS 2021.
The pipeline consists in a cascade of an object detector followed by an affordance detector.
object_detector is SSDLite MobileNetV3 Small trained on IIT-AFF dataset.
affordance_detector is MobileNetV1-Unet trained on IIT-AFF dataset. It is loaded through the keras segmentation library.
"""
import six
import tensorflow as tf
import os
import cv2
import numpy as np

from keras_segmentation import predict
from keras_segmentation.data_utils.data_loader import DataLoaderError


def map_classes(mask):
    """ Returns the mapped affordance mask.
    affordance_detector was originally trained to distinguish 4 classes: Background (0), Wrap-grasp (1), No-grasp (2) and Grasp (3).
    In the paper we consider only Background (0), Grasp (1) and No-grasp (2).
    """
    mask[mask == 1] = 1  # Wrap-grasp
    mask[mask == 2] = 2  # No-grasp
    mask[mask == 3] = 1  # Grasp
    return mask


def image_resize_no_dist(image, width=None, height=None, rotate=False, inter=cv2.INTER_AREA):
    """Returns the resized image without distorting it using zero-padding.

    :param image: image in numpy format
    :param width: final width
    :param height: final height
    :param rotate: whether to rotate the image if it has horizontal shape or not
    :param inter: type of interpolation
    :return: resized and padded image avoiding distorsion
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    if (w > h) and rotate:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    # if width is None:
    # calculate the ratio of the height and construct the
    # dimensions
    r = height / float(h)
    if (int(w * r) < width):
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    (h, w) = resized.shape[:2]

    delta_w = width - w
    delta_h = height - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    # return the resized image
    return padded


def get_image_array(image_input,
                    width, height,
                    imgNorm="sub_mean", ordering='channels_first'):
    """ Load image array from input """

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_image_array: path {0} doesn't exist"
                                  .format(image_input))
        img = cv2.imread(image_input, 1)
    else:
        raise DataLoaderError("get_image_array: Can't process input type {0}"
                              .format(str(type(image_input))))

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = image_resize_no_dist(img, width, height, rotate=True, inter=cv2.INTER_AREA)
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img / 255.0

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


# Load video
cap = cv2.VideoCapture(os.path.join(os.curdir, "video", "video.mp4"))

# Load object detector and build the detection function
od_model = tf.saved_model.load(os.path.join(os.curdir, "object_detector", "saved_model"))
detect_fn = od_model.signatures['serving_default']
od_input_width = 320
od_input_height = 320

# Load affordance detection model
aff_model = predict.model_from_checkpoint_path(os.path.join(os.curdir, "affordance_detector", "mobilenet_unet_p_1"))
aff_model.summary()
aff_input_width = 224
aff_input_height = 224
aff_output_width = 112
aff_output_height = 112
n_classes = 4

# Inference loop
while cap.isOpened():
    # Load frame
    ret, frame = cap.read()

    # Convert to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.imshow('RGB', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Resize without distorsion
    image_np = image_resize_no_dist(image, od_input_width, od_input_height, rotate=False, inter=cv2.INTER_AREA)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Object detection
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    od_prediction = image_np.copy()
    for i, box in enumerate(detections['detection_boxes']):
        if detections['detection_scores'][i] > 0.7:
            # Values inside box array are normalized
            # box[0] is ymin
            # box[1] is xmin
            # box[2] is ymax
            # box[3] is xmax
            bbox = [int(round(box[1] * od_input_width)), int(round(box[0] * od_input_height)),
                    int(round(box[3] * od_input_width)), int(round(box[2] * od_input_height))]

            # Crop object
            od_output = od_prediction[bbox[1]:bbox[3], bbox[0]:bbox[2]]

            # Draw rectangle
            cv2.rectangle(od_prediction, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255),
                          thickness=2)

            # Affordance pre-processing
            x = get_image_array(od_output, width=aff_input_width, height=aff_input_height, ordering="channels_last")

            # Affordance prediction
            aff_prediction = aff_model.predict(np.array([x]))[0]
            aff_prediction = aff_prediction.reshape((aff_output_height, aff_output_width, n_classes)).argmax(axis=2)

            aff_prediction = map_classes(aff_prediction.astype(np.uint8))

            # Visualize colored affordance
            black = np.zeros((aff_prediction.shape[0], aff_prediction.shape[1], 3))
            black[aff_prediction == 1, :] = [0, 0, 255]  # blue = grasp
            black[aff_prediction == 2, :] = [0, 255, 0]  # verde = non grasp
            (h, w) = od_output.shape[:2]
            if w > h:
                black = cv2.rotate(black, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.namedWindow("Affordance detection", cv2.WINDOW_NORMAL)
            cv2.imshow("Affordance detection", cv2.cvtColor(black.astype(np.uint8), cv2.COLOR_RGB2BGR))
    # Visualize object detection
    cv2.namedWindow("Object detection", cv2.WINDOW_NORMAL)
    cv2.imshow('Object detection', cv2.cvtColor(od_prediction, cv2.COLOR_RGB2BGR))
    cv2.waitKey(10)

cap.release()
cv2.destroyAllWindows()
