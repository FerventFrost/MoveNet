'''
TensorFlow Tutorial for Movenet: 
    https://www.tensorflow.org/hub/tutorials/movenet
github Project:
    https://github.com/FerventFrost/MoveNet
'''

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2

import UtilityFunctions as UTIF


def movenet(inputImage, ImportedModel):
    '''Run Movenet model. 

    Args:
        InputImage: is a tf imported image, use tf.io.read_file() to read image and then use tf.image.decode_x() to decode to desired type.
        ImportedModel: the model that is imported using LoadModel() function.

    Returns:
        A numpy array of keypoints of body's joints'''

    model = ImportedModel.signatures['serving_default']

    inputImage = tf.cast(inputImage, dtype=tf.int32)
    outputs = model(inputImage)
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores


if __name__ == "__main__":

    # Download Model and Load
    ModelURL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    input_size = 192
    Imported = UTIF.LoadModel(ModelURL)

    # Read Image, Decode, and Resize it
    ImagePath = "Test/P2.jpg"
    Image = tf.io.read_file(ImagePath)
    Image = tf.image.decode_jpeg(Image)
    InputImage = tf.expand_dims(Image, axis=0)
    InputImage = tf.image.resize_with_pad(InputImage, input_size, input_size)

    # Run Model
    keypoints_with_scores = movenet(InputImage, Imported)

    # Draw Body's Joint Point
    display_image = tf.expand_dims(Image, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(
    display_image, 1280, 1280), dtype=tf.int32)
    output_overlay = UTIF.draw_prediction_on_image(
        np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

    #Display Result
    Results = cv2.cvtColor(output_overlay, cv2.COLOR_RGB2BGR)
    cv2.imshow("Results",Results)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
