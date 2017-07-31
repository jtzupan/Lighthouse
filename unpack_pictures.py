import json
import urllib.request
import os
from scipy.misc import imread
import dlib
import numpy as np
import glob

# with open('data.json') as file_name:
#     data = json.load(file_name)

# for i in range(0, len(data['results'])):
#     try:
#         picture_URL = data['results'][i]['officialPictureUrl']
#         name = data['results'][i]['displayName']
#         with urllib.request.urlopen(picture_URL) as url:
#             with open('Mugshots/{}.jpg'.format(name), 'wb') as f:
#                 f.write(url.read())
#
#     except KeyError:
#         pass

# TODO: process the photos to store the encodings

face_recognition_model_loc = "flask/Models/dlib_face_recognition_resnet_model_v1.dat"
face_encoder = dlib.face_recognition_model_v1(face_recognition_model_loc)
face_detector = dlib.get_frontal_face_detector()
shape_predictor_model_loc = "flask/Models/shape_predictor_68_face_landmarks.dat"
pose_predictor = dlib.shape_predictor(shape_predictor_model_loc)


def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate,
        but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimentional face encodings (one for each face in the image)
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations)

    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters))
            for raw_landmark_set in raw_landmarks]

def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)


def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def load_image_file(filename, mode='RGB'):
    """
    Loads an image file (.jpg, .png, etc) into a numpy array

    :param filename: image file to load
    :param mode: format to convert the image to. Only 'RGB' (8-bit RGB, 3 channels) and 'L' (black and white)
        are supported.
    :return: image contents as numpy array
    """
    return imread(filename, mode=mode)


def _raw_face_locations(img, number_of_times_to_upsample=1):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find
        smaller faces.
    :return: A list of dlib 'rect' objects of found face locations
    """
    return face_detector(img, number_of_times_to_upsample)


def face_locations(img, number_of_times_to_upsample=1):
    """
    Returns an array of bounding boxes of human faces in a image

    :param img: An image (as a numpy array)
    :param number_of_times_to_upsample: How many times to upsample the image looking for faces. Higher numbers find
        smaller faces.
    :return: A list of tuples of found face locations in css (top, right, bottom, left) order
    """
    return [_trim_css_to_bounds(_rect_to_css(face), img.shape)
            for face in _raw_face_locations(img, number_of_times_to_upsample)]


def _raw_face_landmarks(face_image, face_locations=None):
    if face_locations is None:
        face_locations = _raw_face_locations(face_image)
    else:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]

    return [pose_predictor(face_image, face_location) for face_location in face_locations]


def face_landmarks(face_image, face_locations=None):
    """
    Given an image, returns a dict of face feature locations (eyes, nose, etc) for each face in the image

    :param face_image: image to search
    :param face_locations: Optionally provide a list of face locations to check.
    :return: A list of dicts of face feature locations (eyes, nose, etc)
    """
    landmarks = _raw_face_landmarks(face_image, face_locations)
    landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]

    # For a definition of each point index, see https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png
    return [{
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] + [points[64]]
    } for points in landmarks_as_tuples]


def face_encodings(face_image, known_face_locations=None, num_jitters=1):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate,
        but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimentional face encodings (one for each face in the image)
    """
    raw_landmarks = _raw_face_landmarks(face_image, known_face_locations)

    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters))
            for raw_landmark_set in raw_landmarks]


# convert the photos to encodings and save to the disk
names = []
encodings = []
for filename in glob.glob("flask/Mugshots/*.jpg"):
    # print(filename)
    names.append(filename[11:-4])
    encodings.append(face_encodings(imread(filename, mode='RGB'))[0])

# print(type(encodings[0]))
for i, encoding in enumerate(encodings):
    np.save(file="encodings/{}.npy".format(names[i]), arr=encoding)  # exception raised
