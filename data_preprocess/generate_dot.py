import multiprocessing
import os
import json
import shutil
import time
import pandas as pd
from joblib import Parallel, delayed

input_path = "./bigvul/processed_train/"
bin_output_path = "./bigvul/all_bin/processed_train/"
dot_output_path = "./bigvul/all_dot/processed_train/"

if not os.path.exists(bin_output_path):
    os.makedirs(bin_output_path)

if not os.path.exists(dot_output_path):
    os.makedirs(dot_output_path)


def joern_parse(idx):
    """
    idx: 0,1,2,...
    """
    filelist = os.listdir(input_path)
    filename = filelist[idx]
    if filename in no_process_list:
        print(f"{filename} donot process")
        return

    folder_path = os.path.join(input_path, filename)
    bin_name = filename + ".bin"
    bin_filepath = os.path.join(bin_output_path, bin_name)
    dot_dir = os.path.join(dot_output_path, filename)

    cmd = f"joern-parse {folder_path} -o {bin_filepath}"
    print(cmd)
    os.system(cmd)

    cmd = f"joern-export {bin_filepath} --repr all --format dot --out {dot_dir}"
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    filelist = os.listdir(input_path)
    # 获取CPU数量（逻辑核心数）
    num_cpus = multiprocessing.cpu_count()
    print(num_cpus)
    for i in range(len(filelist)):
        joern_parse(i)
    # Parallel(n_jobs=num_cpus, prefer='threads')(delayed(joern_parse)(i) for i in range(len(filelist)))


