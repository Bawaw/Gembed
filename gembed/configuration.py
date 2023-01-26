#!/usr/bin/env python3

import configparser
import os

from gembed.definitions import ROOT_DIR


class Configuration:
    def __init__(self):
         self._config = configparser.ConfigParser()
         self._config.read(
             os.path.join(ROOT_DIR, 'conf/configuration.conf')
         )

    def __getitem__(self, item):
        return self._config[item]


# TEST SCRIPT
if __name__ == "__main__":
    configuration = Configuration()
    print()
