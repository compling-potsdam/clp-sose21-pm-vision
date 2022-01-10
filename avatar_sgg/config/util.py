import yaml
import os

def get_config():
    """
    Get the general project config
    important dictionary keys:
    "game_setup", ""ade20k" for the location of the dataset with captions
    :return:
    """
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_file, "r") as read_file:
        conf = yaml.load(read_file, Loader=yaml.FullLoader)

    if conf["use_dev_config"]:
        print("Dev Setup: dev_config.yaml will be used")
        del conf
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev_config.yaml")
        with open(config_file, "r") as read_file:
            conf = yaml.load(read_file, Loader=yaml.FullLoader)
    else:
        print("Server Setup: config.yaml will be used")
    return conf
if __name__ == "__main__":
    print("Start")
    conf = get_config()

    print("Done")