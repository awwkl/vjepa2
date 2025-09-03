import os
import subprocess

# overall_dir = 'pretrain'
overall_dir = 'evals/vitl'

upload_path = f"gs://ccwm/ccwm/models/vjepa2/babyview/{overall_dir}"
command = ["gsutil", "cp", "-r", overall_dir, upload_path]
subprocess.run(command)

# for model_dir in model_dir_list:
#     assert os.path.exists(model_dir), f"Model path {model_dir} does not exist."
    
#     # find all files that end with .pt inside the dir
#     model_path_list = [
#         os.path.join(model_dir, f)
#         for f in os.listdir(model_dir)
#         # if (f.endswith(".pt") or f.endswith(".yaml") or f.endswith(".csv"))
#     ]

#     for model_path in sorted(model_path_list):
#         breakpoint()
#         upload_path = f"gs://ccwm/ccwm/models/vjepa2/babyview/{model_path}"
#         command = ["gsutil", "cp", model_path, upload_path]
#         subprocess.run(command)
