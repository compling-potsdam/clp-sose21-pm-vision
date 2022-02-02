from avatar_sgg.config.util import get_config
import collections
import pandas as pd
import string
import json
import random
import torch
import torch.utils.data as data
import os


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
    sequences_df["d1"] = sequences_df["d1"].map(lambda a: a if a[-1] in string.punctuation else a + ". ")
    sequences_df["d2"] = sequences_df["d2"].map(lambda a: a if a[-1] in string.punctuation else a + ". ")
    sequences_df["d3"] = sequences_df["d3"].map(lambda a: a if a[-1] in string.punctuation else a + ". ")
    sequences_df["d4"] = sequences_df["d4"].map(lambda a: a if a[-1] in string.punctuation else a + ". ")
    sequences_df["d5"] = sequences_df["d5"].map(lambda a: a if a[-1] in string.punctuation else a + ". ")
    sequences_df["merged_sequences"] = sequences_df[["d1", "d2", "d3", "d4", "d5"]].agg(lambda x: ''.join(x.values),
                                                                                        axis=1).T
    sequences_fram = sequences_df[["image_id", "image_path", "image_cat", "merged_sequences"]]
    captions_df = pd.merge(captions_df, sequences_fram, how='inner', left_on=['image_id'], right_on=['image_id'])
    captions_df["image_path"] = captions_df["image_path"].map(
        lambda a: os.path.join("file://", ade20k_dir, "images", a))
    captions_df.drop(["Unnamed: 0"], axis=1)

    captions_list = [{"image_id": row["image_id"], "id": row["caption_id"], "caption": row["caption"],
                      "image_path": row["image_path"], "image_cat": row["image_cat"],
                      "merged_sequences": row["merged_sequences"]} for i, row in captions_df.iterrows()]
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
            image_path_to_caption[image_path]["caption"] = [caption]
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
    cat = {}
    if "category" in split[0].keys():
        cat = {i: split[k]["category"] for i, k in enumerate(split)}
    return cat

def group_entry_per_category(category):
    category_to_entry_lookup = collections.defaultdict(list)
    for k, v in category.items():
        category_to_entry_lookup[v].append(k)


class SceneGraphDataset(data.Dataset):
    """ SGEncoding dataset """

    def __init__(self, train_ids, test_ids, test_on=False, val_on=False, num_test=5000, num_val=5000):
        super(SceneGraphDataset, self).__init__()

        conf = get_config()["scene_graph"]
        cap_graph_file = conf["capgraphs_file"]
        vg_dict_file = conf["visual_genome_dict_file"]
        image_data_file = conf["image_data"]
        image_dir = conf["image_dir"]

        self.cap_graph = json.load(open(cap_graph_file))
        vg_dict = json.load(open(vg_dict_file))
        self.image_data = json.load(open(image_data_file))


        self.coco_ids_to_image_path = {
            str(self.image_data[i]["coco_id"]): os.path.join(image_dir, str(self.image_data[i]["image_id"]) + "jpg") for
        i, entry
            in enumerate(self.cap_graph["vg_coco_ids"]) if entry > -1}

        # self.coco_image_data = [str(self.image_data[i]) for i, entry in enumerate(self.cap_graph["vg_coco_ids"]) if
        #                         entry > -1]
        # Warning somehow, not the same size...
        # num_good_paths = sum([1 for entry in self.coco_image_data if
        #      os.path.exists(image_dir + str(entry["image_id"]) + ".jpg")])
        # num_coco_ids = len(cap_graph['vg_coco_id_to_caps'])
        # assert num_good_paths == num_coco_ids

        self.train_ids = train_ids
        self.test_ids = test_ids
        if test_on:
            self.key_list = self.test_ids[:num_test]
        elif val_on:
            self.key_list = self.test_ids[num_test:num_test + num_val]
        else:
            self.key_list = self.test_ids[num_test + num_val:] + self.train_ids

    def __getitem__(self, item):

        coco_id = self.key_list[item]
        return self.coco_ids_to_image_path[coco_id], self.cap_graph["vg_coco_id_to_caps"][coco_id]

    def __len__(self):
        return len(self.key_list)


class SimpleCollator(object):
    def __call__(self, batch):

        glue = {path:{"captions": captions} for path, captions in batch}

        return glue


def get_scene_graph_splits():
    """
    Get the training split used for Image Retrieval using Scence Graph
    from https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/
    :return: a tuple with training and test ids (intersection between Visual Genome and MSCOCO), and the linked data
    """
    conf = get_config()
    output_dir = conf["output_dir"]
    conf = conf["scene_graph"]

    train_id_file = os.path.join(output_dir, 'train_ids.json')
    test_id_file = os.path.join(output_dir, 'test_ids.json')

    ids_created = os.path.exists(train_id_file)

    if not ids_created:

        sg_train_path = conf["train"]
        sg_test_path = conf["test"]
        sg_val_path = conf["val"]
        print("Loading samples. This can take a while.")
        sg_data_train = json.load(open(sg_train_path))
        sg_data_val = json.load(open(sg_val_path))
        sg_data_test = json.load(open(sg_test_path))
        # sg_data = torch.load(sg_train_path)
        # sg_data.update(torch.load(sg_test_path))
        # Merge the val sample to the training data, it would be a waste...
        sg_data_train.update(sg_data_val)
        sg_data = sg_data_train.copy()
        sg_data.update(sg_data_test)
        train_ids = list(sg_data_train.keys())
        test_ids = list(sg_data_test.keys())

        with open(train_id_file, 'w') as f:
            json.dump(train_ids, f)
        print("created:", train_id_file)
        with open(test_id_file, 'w') as f:
            json.dump(test_ids, f)
        print("created:", test_id_file)
    else:
        train_ids = json.load(open(train_id_file))
        test_ids = json.load(open(test_id_file))

    print("Number of Training Samples", len(train_ids))
    print("Number of Testing Samples", len(test_ids))
    return train_ids, test_ids


def get_scene_graph_loader(batch_size, train_ids, test_ids, test_on=False, val_on=False, num_test=5000, num_val=1000):
    """ Returns a data loader for the desired split """
    split = SceneGraphDataset(train_ids, test_ids, test_on=test_on, val_on=val_on, num_test=num_test,
                              num_val=num_val)

    loader = torch.utils.data.DataLoader(split,
                                         batch_size=batch_size,
                                         shuffle=not (test_on or val_on),  # only shuffle the data in training
                                         pin_memory=True,
                                         # num_workers=4,
                                         collate_fn=SimpleCollator(),
                                         )
    return loader


if __name__ == "__main__":
    print("Start")
    batch_size = 100
    # train, dev, test = get_ade20k_split()
    # print(f"Train Split: {len(train)}")
    # print(f"Dev Split: {len(dev)}")
    # print(f"Test Split: {len(test)}")
    train_ids, test_ids = get_scene_graph_splits()

    use_test = True
    use_val = False
    loader = get_scene_graph_loader(batch_size, train_ids, test_ids, test_on=use_test, val_on=use_val)
    next(enumerate(loader))
    print("Done")
