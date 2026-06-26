Grasp Label Generator

Usage:
  ./run.sh --obj_path xxx.obj --unit mm --sparse_num 3000 --max_width 0.1

Or:
  CUDA_VISIBLE_DEVICES=0 python gen_sparse_label.py --obj_path xxx.obj --unit mm

Requirements:
  pip install torch numba numpy open3d>=0.16.0 tqdm
