import yaml


def config_parse(file_path):
    return yaml.full_load(open(file_path))


if __name__ == '__main__':
    config = config_parse('config.yaml')
    print(config)
