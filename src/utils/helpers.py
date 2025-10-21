import os

def create_dirs():
    dirs = ["data", "data/raw", "data/processed"]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)