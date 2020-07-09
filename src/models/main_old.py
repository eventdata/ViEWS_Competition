'''Main file to run the program
Author: Yi-Fan Li
Date: Jun 1, 2020
'''

# Add the parent folder into the pythonpath
import sys, os
sys.path.append(os.path.abspath(os.curdir))

import argparse
from src.data.data_utils import get_adj_matrix



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STGCN_ViEWS')

print('hello')