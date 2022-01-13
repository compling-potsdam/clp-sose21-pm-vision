from avatar_sgg.config.util import get_config
import os
import collections
import pandas as pd
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def get_ade20k_caption_annotations():
    """
    Precondition: checkout the https://github.com/clp-research/image-description-sequences under the location
    of the ade20k_dir directoiry
    :return:
    """
    conf = get_config()["ade20k"]

    ade20k_dir = conf["root_dir"]
    ade20k_caption_dir = conf["caption_dir"]
    captions_file = os.path.join(ade20k_caption_dir, "captions.csv")
    sequences_file = os.path.join(ade20k_caption_dir, "sequences.csv")
    captions_df = pd.read_csv(captions_file, sep="\t", header=0)
    sequences_df = pd.read_csv(sequences_file, sep="\t", header=0)
    sequences_fram = sequences_df[["image_id", "image_path"]]
    captions_df = pd.merge(captions_df, sequences_fram, how='inner', left_on=['image_id'], right_on=['image_id'])
    captions_df.image_path = captions_df.image_path.map(lambda a: os.path.join("file://", ade20k_dir, "images", a))
    captions_df.drop(["Unnamed: 0"], axis=1)

    captions_list = [{"image_id": row["image_id"], "id": row["caption_id"], "caption": row["caption"],
                      "image_path": row["image_path"]} for i, row in captions_df.iterrows()]
    # { id: list(captions_df[captions_df["image_id"] == id ]["caption"]) for id in ids  }

    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in captions_list:
        caption = val['caption']
        image_path = val["image_path"]
        image_path_to_caption[image_path].append(caption)

    return image_path_to_caption


def get_ade20k_split(test_proportion: int = 15, test_size: int = 10):
    """
    Returns train, dev and test split.
    Dev has only one image.
    TODO: probably better to use cross validation for the splits
    :param test_proportion:
    :return:
    """
    assert test_proportion > 0 and test_proportion < 100
    captions = get_ade20k_caption_annotations()
    # Make the split consistent
    random.seed(1)
    keys = list(captions.keys())
    random.shuffle(keys)
    start_idx = test_size
    dev = {k: captions[k] for k in keys[:test_size]}
    size = len(keys[start_idx:])
    test_idx = int(test_proportion * size / 100)
    test = {k: captions[k] for k in keys[start_idx:test_idx]}
    train = {k: captions[k] for k in keys[test_idx:]}
    return train, dev, test

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

if __name__ == "__main__":
    print("Start")
    train, dev, test = get_ade20k_split()
    print("Done")
