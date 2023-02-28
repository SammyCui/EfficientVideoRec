import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)

args = parser.parse_args()
print(args.name)