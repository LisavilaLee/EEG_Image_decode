import os
from huggingface_hub import hf_hub_download

# 设置 Hugging Face 镜像站点
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"

# 资源信息
repo_id = "LidongYang/EEG_Image_decode"
file_path = "test_image_latent_512.pt"
target_dir = "/userhome2/liweile/EEG_Image_decode/"

# 下载文件到指定目录
file_path_local = hf_hub_download(
    repo_id=repo_id,
    filename=file_path,
    cache_dir=target_dir,
    repo_type="dataset"
)

print(f"文件已下载至：{file_path_local}")