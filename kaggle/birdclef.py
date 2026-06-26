import os
import shutil
import kagglehub

kagglehub.login()

# # 1. Define your custom path
# target_dir = "../data"

# cache_path = kagglehub.competition_download('birdclef-20-26')

# if not os.path.exists(target_dir):
#     shutil.copytree(cache_path, target_dir)
#     print(f"Data successfully copied to: {target_dir}")
# else:
#     print(f"Target directory {target_dir} already exists!")