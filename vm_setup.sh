export WANDB_API_KEY=6ee6df69a17811e60b402f0f1564bd11ec4fd2eb
sudo apt-get update -y
sudo apt-get install git -y
sudo apt install python3.10-virtualenv -y
python3 -m venv .venv
source .venv/bin/activate
SYSTEM_VERSION_COMPAT=0 pip install dmlab2d
pip install -e . --no-cache-dir
sh ray_patch.sh
pip install jax==0.4.14 jaxlib==0.4.14
sudo apt-get install ffmpeg libsm6 libxext6  -y