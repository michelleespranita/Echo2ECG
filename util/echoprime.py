import torch
import cv2

VIEW_TYPES = [
    'A2C',
    'A3C',
    'A4C',
    'A5C',
    'Apical_Doppler',
    'Doppler_Parasternal_Long',
    'Doppler_Parasternal_Short',
    'Parasternal_Long',
    'Parasternal_Short',
    'SSN',
    'Subcostal'
 ]
NUM_VIEWS = len(VIEW_TYPES)

ECHOPRIME_MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
ECHOPRIME_STD = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)

def crop_and_scale(img, res=(224, 224), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]
    r_out = res[0] / res[1]

    if r_in > r_out:
        padding = int(round((in_res[0] - r_out * in_res[1]) / 2))
        img = img[:, padding:-padding]
    if r_in < r_out:
        padding = int(round((in_res[1] - in_res[0] / r_out) / 2))
        img = img[padding:-padding]
    if zoom != 0:
        pad_x = round(int(img.shape[1] * zoom))
        pad_y = round(int(img.shape[0] * zoom))
        img = img[pad_y:-pad_y, pad_x:-pad_x]

    img = cv2.resize(img, res, interpolation=interpolation)
    return img