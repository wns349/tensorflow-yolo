from net.yolov2 import YoloV2


def _update_configs(configs, configs_path):
    for k, v in configs.items():
        if k.endswith("_dir"):
            if not os.path.isabs(v):
                configs[k] = os.path.join(os.path.dirname(os.path.abspath(configs_path)), v);
    return configs


def _main(cfg, mode):
    version = cfg["COMMON"]["version"]
    if version == "v2":
        yolo = YoloV2()
    else:
        raise ValueError("Unsupported version: {}".format(version))

    if mode == "train":
        pass
    elif mode == "test":
        pass
    elif mode == "anchor":
        params = {**cfg["ANCHOR"], **cfg["COMMON"]}
        anchors, class_names = yolo.generate_anchors(params)
        print("Anchors: ")
        print("\t{}".format(anchors))
        print("Class names: ")
        print("\t{}".format(class_names))
    else:
        raise ValueError("Unsupported mode: {}".format(mode))


if __name__ == "__main__":
    import argparse
    import configparser
    import os

    args = argparse.ArgumentParser()
    args.add_argument("--config", dest="config", help="Path to configuration file",
                      default=os.path.join(os.path.dirname(__file__), "config", "yolo_2.ini"))
    args.add_argument("--mode", dest="mode", help="Mode: (train|test|anchor)",
                      default="anchor")
    c = args.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(c.config)
    cfg = {s: _update_configs(dict(cfg.items(s)), c.config) for s in cfg.sections()}

    _main(cfg, c.mode)
