#
# Created on Thu Aug 24 2023
#
# Copyright (c) 2023 The Home Made AI (HOMAI)
# Author: Javad Rezaie
# License: Apache License 2.0
#

from mmengine.config import Config
from mmengine.runner import Runner
import argparse

def main(args):

    config = Config.fromfile(args.config_path)
    config.launcher = "pytorch"
    runner = Runner.from_cfg(config)
    runner.train()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get Config Path.')
    parser.add_argument('config_path', type=str, help='path to the config file')
    args = parser.parse_args()
    main(args)
