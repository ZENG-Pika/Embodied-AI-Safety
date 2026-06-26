set -ex

export IMAGEIO_FFMPEG_EXE=ffmpeg
export OMP_NUM_THREADS=128

export PYTHONPATH=YOUR_PATH/openpi/src:YOUR_PATH/openpi/packages/openpi-client/src:YOUR_PATH/openpi/third_party/lerobot:${PYTHONPATH}
conda activate pi0

cd YOUR_PATH/openpi
ulimit -n 1000000
config_name=$1
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9  python scripts/train.py ${config_name}  \
  --exp-name=${config_name}