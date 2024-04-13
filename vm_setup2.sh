sudo apt install python3.10-virtualenv -y
python3 -m venv .venv
source .venv/bin/activate
SYSTEM_VERSION_COMPAT=0 pip install dmlab2d
pip install -e . --no-cache-dir
sh ray_patch.sh
pip install jax==0.4.14 jaxlib==0.4.14
sudo apt-get install ffmpeg libsm6 libxext6  -y