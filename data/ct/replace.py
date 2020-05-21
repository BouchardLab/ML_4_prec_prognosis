import sys
import re
import numpy as np
import argparse

def check_token(t):
    if fp_re.match(t):
        return str(np.random.standard_normal())[:len(t)]
    return t

parser = argparse.ArgumentParser(description='replace real CT measures with fake data and print results to stdout')
parser.add_argument('ct_measure_file', help='the CT measure file to replace with fake data')

args = parser.parse_args()

np.random.seed(1)
fp_re = re.compile(r'([-]?\d+\.\d+)')

with open(sys.argv[1], 'r') as f:
    for line in f:
        sys.stdout.write("".join(map(check_token, fp_re.split(line))))
