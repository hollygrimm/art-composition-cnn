import argparse
import json
import os
import time

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def get_config_from_json(json_file):
    """Get the config from a json file

    Args:
        json_file (str): path

    Returns:
        config (dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)
    return config_dict

def process_config(json_file):
    config_dict = get_config_from_json(json_file)
    # TODO: add time to path
    log_dir = os.path.join('experiments', time.strftime("%Y-%m-%d/",time.localtime()), config_dict['exp_name'], 'logs/')
    checkpoint_dir = os.path.join('experiments', time.strftime("%Y-%m-%d/",time.localtime()), config_dict['exp_name'], 'checkpoints/')
    return config_dict, log_dir, checkpoint_dir

def create_dirs(dirs):
    """Create directories if not found

    Args:
        dirs (str): directories

    Returns:
        exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
    exit(-1)