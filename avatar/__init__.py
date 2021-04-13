import os
import json
import zipfile


def load_project_resource(file_path: str):
    """
    Tries to load a resource:
        1. directly
        2. from the egg zip file
        3. from the egg directory

    This is necessary, because the files are bundled with the project.

    :return: the file as json
    """
    ...
    if not os.path.isfile(file_path):
        try:
            egg_path = __file__.split(".egg")[0] + ".egg"
            if os.path.isfile(egg_path):
                print(f"Try to load instances from ZIP at {egg_path}")
                with zipfile.ZipFile(egg_path) as z:
                    f = z.open(file_path)
                    data = json.load(f)
            else:
                print(f"Try to load instances from directory at {egg_path}")
                with open(egg_path + '/' + file_path) as f:
                    data = json.load(f)
        except Exception:
            raise FileNotFoundError(f"Could not find '{file_path}'. "
                                    "Make sure you run the script from the correct directory.")
    else:
        with open(file_path) as f:
            data = json.load(f)
    return data
