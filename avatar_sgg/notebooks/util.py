import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def resize_img_to_array(img, img_shape=(244, 244)):
    """
    REshape an Image to the given Size
    :param img: Image as Numpy array
    :param img_shape: The new shape of the image, tuple of height and width
    :return:
    """
    # Code from https://gist.github.com/yinleon/8a6bf8c2f3226a521680f930f3d4339f
    img_array = np.array(
        img.resize(
            img_shape,
            Image.ANTIALIAS
        )
    )

    return img_array

def image_grid(fn_images: list,
               text: list = [],
               top: int = 12,
               per_row: int = 2,
               dim: tuple = (150, 150)):
    """
    # Code from https://gist.github.com/yinleon/8a6bf8c2f3226a521680f930f3d4339f
    fn_images is a list of image paths.
    text is a list of annotations.
    top is how many images you want to display
    per_row is the number of images to show per row.
    """
    img_width, img_height = dim
    for i in range(len(fn_images[:top])):
        if i % per_row == 0:
            _, ax = plt.subplots(1, per_row,
                                 sharex='col',
                                 sharey='row',
                                 figsize=(15, 5))
        j = i % per_row
        image = Image.open(fn_images[i])
        image = resize_img_to_array(image,
                                    img_shape=(img_width,
                                               img_height))
        ax[j].imshow(image)
        ax[j].axis('off')
        if text:
            offset = -10
            start = -30
            for t in text[i]:
                start += offset
                ax[j].annotate(t,
                               (0, 0), (0, start),
                               xycoords='axes fraction',
                               textcoords='offset points',
                               va='top')
