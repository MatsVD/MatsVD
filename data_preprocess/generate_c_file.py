import os
import shutil
import pandas as pd
import numpy as np

import os
import pandas as pd
from tqdm import tqdm

file_path = "processed_val.csv"

df_all = pd.read_csv(file_path)

df_vul = df_all[df_all["function_label"] == 1].reset_index(drop=True)

df_non_vul = df_all[df_all["function_label"] == 0].reset_index(drop=True)

df = pd.concat((df_vul, df_non_vul))
df = df.sample(frac=1).reset_index(drop=True)  # 对DataFrame进行了完全的洗牌,重新设置了每行的index

print("\n*******\n", f"total non-vul funcs in  data: {len(df_non_vul)}")
print(f"total vul funcs in  data: {len(df_vul)}", "\n*******\n")


labels = df["statement_label"].tolist()
source = df["func_before"].tolist()
indexs = df["index"].tolist()

for i in tqdm(range(len(source))):
    func_index = indexs[i]
    func = source[i]
    print(func_index)

    filedir = os.path.join("bigvul/processed_val", str(func_index))
    file_name = f"{func_index}.c"
    if os.path.exists(filedir):
        shutil.rmtree(filedir)
    os.makedirs(filedir)
    with open(os.path.join(filedir, file_name), 'w', encoding='utf-8') as f:
        f.write(func)






# num = 0
# for file in vullist:
#     if file not in vulp:
#         source_folder = os.path.join(vuldir, file)
#         destination_folder = os.path.join("bigvul/wait_to_process/", file)
#         shutil.copytree(source_folder, destination_folder)
#         num += 1
#         if num == 727:
#             break

# vul_processed = "bigvul/vul_processed"
# vulp = os.listdir(vul_processed)
#
# inputdir = "bigvul/train/node_json/"
# l = os.listdir(inputdir)
#
# for file in l:
#     name = file.split("_")[0]
#     if name in vulp:
#         source_file = os.path.join(inputdir, file)
#         destination_file = os.path.join("bigvul/", file)
#         shutil.move(source_file, destination_file)



