#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('OpenBMB/VoxCPM2')
print(model_dir)