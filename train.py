import argparse
import yaml


def args_parse():
    p = argparse.ArgumentParser()

    # get config.yaml file path
    p.add_argument("--config_path", required=True, help="config_yaml_file_path[ex) ./config.yaml]")

    return p.parse_args()


def load_config(path):
    """ load train config """
    with open(path) as f:
        y = yaml.safe_load(f)

    return argparse.Namespace(**y)


def main(config):
    pass


if __name__ == "__main__":
    args = args_parse()
    conf = load_config(args.config_path)
    main(conf)