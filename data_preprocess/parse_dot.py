import copy
import multiprocessing
import os
import json
import shutil
import time
import pandas as pd
from joblib import Parallel, delayed

import networkx


dot_output_path = "./joern_wc/bigvul/all_dot/processed_val"
edgejson_dir = "./joern_wc/bigvul/edge_json/processed_val"

if not os.path.exists(edgejson_dir):
    os.makedirs(edgejson_dir)

def generate_json(idx):
    filename = empty_edgejson_func_index[idx]
    folder_path = os.path.join(dot_output_path,filename)
    dotfile_path = os.path.join(folder_path,"export.dot")
    print(dotfile_path)

    try:
        G = networkx.drawing.nx_pydot.read_dot(dotfile_path)
        new_G = copy.deepcopy(G)

        delete_nodes_label = ['METHOD']

        reserved_nodes = []
        node_id_to_lineno = {}
        for node_id, key_value in new_G.nodes.data():
            if 'CODE' in key_value and 'LINE_NUMBER' in key_value and key_value['CODE'] != '"<empty>"' and \
                    key_value['CODE'] != '"<global>"' and key_value['label'] not in delete_nodes_label:
                node_id_to_lineno[node_id] = int(key_value['LINE_NUMBER'])
                reserved_nodes.append(node_id)

        edges = []
        for startnode_id, endnode_id, key_value in new_G.edges.data():
            if startnode_id in reserved_nodes and endnode_id in reserved_nodes and new_G.nodes[startnode_id]['LINE_NUMBER'] != new_G.nodes[endnode_id]['LINE_NUMBER']:
                print(startnode_id, new_G.nodes[startnode_id])
                print(endnode_id, new_G.nodes[endnode_id])
                print(key_value['label'])
                print()

                e = []
                line_num_a = node_id_to_lineno[startnode_id] - 1
                line_num_b = node_id_to_lineno[endnode_id] - 1
                e_type = key_value['label']
                e = [line_num_a, line_num_b, e_type]
                if e not in edges:
                    edges.append(e)


        filepath = os.path.join(edgejson_dir, str(filename) + '_edge.json')
        with open(filepath, 'w') as json_file:
            json.dump(edges, json_file)

    except Exception as e:
        error_message = f"Error processing file {dotfile_path}: {e}\n"
        print(error_message)  # 在控制台打印错误信息
        with open('error_dot.txt', 'a') as file:
            file.write(error_message)  # 将错误信息追加写入到文本文件


if __name__ == '__main__':
    # 获取CPU数量（逻辑核心数）
    num_cpus = multiprocessing.cpu_count()
    print(num_cpus)
    filelist = os.listdir(dot_output_path)
    for i in range(len(filelist)):
        generate_json(i)

    # filelist = os.listdir(dot_output_path)
    # for idx in range(len(filelist)):
    #     generate_json(idx)
    # Parallel(n_jobs=6, prefer='threads')(delayed(generate_json)(i) for i in range(len(filelist)))

    # Parallel(n_jobs=num_cpus, prefer='threads')(delayed(generate_json)(i) for i in range(len(no_edgejson_func_index)))





