from avatar_sgg.config.util import get_config
import os
import collections
import pandas as pd
import random
import string

def get_ade20k_caption_annotations():
    """
    Precondition: checkout the https://github.com/clp-research/image-description-sequences under the location
    of the ade20k_dir directory
    :return: a dictionary containing the paths to the images as keys. Each image has a dictionary with a  "caption" key
    and a "category" key.
    """
    conf = get_config()["ade20k"]

    ade20k_dir = conf["root_dir"]
    ade20k_caption_dir = conf["caption_dir"]
    captions_file = os.path.join(ade20k_caption_dir, "captions.csv")
    sequences_file = os.path.join(ade20k_caption_dir, "sequences.csv")
    captions_df = pd.read_csv(captions_file, sep="\t", header=0)
    sequences_df = pd.read_csv(sequences_file, sep="\t", header=0)
    sequences_df["d1"] = sequences_df["d1"].map(lambda a: a if a[-1] in string.punctuation else a+ ". ")
    sequences_df["d2"] = sequences_df["d2"].map(lambda a: a if a[-1] in string.punctuation else a + ". ")
    sequences_df["d3"] = sequences_df["d3"].map(lambda a: a if a[-1] in string.punctuation else a + ". ")
    sequences_df["d4"] = sequences_df["d4"].map(lambda a: a if a[-1] in string.punctuation else a + ". ")
    sequences_df["d5"] = sequences_df["d5"].map(lambda a: a if a[-1] in string.punctuation else a + ". ")
    sequences_df["merged_sequences"] = sequences_df[["d1", "d2", "d3", "d4", "d5"]].agg(lambda x: ''.join(x.values), axis=1).T
    sequences_fram = sequences_df[["image_id", "image_path", "image_cat", "merged_sequences"]]
    captions_df = pd.merge(captions_df, sequences_fram, how='inner', left_on=['image_id'], right_on=['image_id'])
    captions_df["image_path"] = captions_df["image_path"].map(lambda a: os.path.join("file://", ade20k_dir, "images", a))
    captions_df.drop(["Unnamed: 0"], axis=1)

    captions_list = [{"image_id": row["image_id"], "id": row["caption_id"], "caption": row["caption"],
                      "image_path": row["image_path"], "image_cat": row["image_cat"], "merged_sequences": row["merged_sequences"]} for i, row in captions_df.iterrows()]
    # { id: list(captions_df[captions_df["image_id"] == id ]["caption"]) for id in ids  }

    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(dict)
    for val in captions_list:
        caption = val['caption']
        category = val['image_cat']
        image_path = val["image_path"]
        merged_sequences = val["merged_sequences"]
        image_path_to_caption[image_path]["category"] = category
        image_path_to_caption[image_path]["merged_sequences"] = merged_sequences
        if "caption" not in image_path_to_caption[image_path].keys():
            image_path_to_caption[image_path]["caption"] = [caption ]
        else:
            image_path_to_caption[image_path]["caption"].append(caption)

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

def get_categories(split):
    return {i:split[k]["category"] for i, k in enumerate(split)}
def group_entry_per_category(category):
    category_to_entry_lookup = collections.defaultdict(list)
    for k, v in category.items():
        category_to_entry_lookup[v].append(k)


if __name__ == "__main__":
    print("Start")
    train, dev, test = get_ade20k_split()
    print(f"Train Split: {len(train)}")
    print(f"Dev Split: {len(dev)}")
    print(f"Test Split: {len(test)}")
    print("Done")
