import os
import cv2
from pathlib import Path
from tqdm import tqdm
def extract_last_frame_from_videos(root_dir, output_dir, xx_last_frame=1):
    """
    遍历目录，找到所有images.rgb.hand_right视频文件，提取最后一帧并保存
    """
    # 查找所有mp4视频文件
    video_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:

            if file.endswith('.mp4') and 'observation/head' in root:
                video_files.append(os.path.join(root, file))
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 处理每个视频文件
    for video_path in tqdm(video_files):
        try:
            # 提取set名称和episode名称
            path_parts = Path(video_path).parts
            set_name = None
            episode_name = None
            for part in path_parts:
                if part.startswith('set'):
                    set_name = part
                if part.startswith('000'):
                    episode_name = part.replace('.mp4', '')
            
            if not set_name or not episode_name:
                print(f"无法从路径中提取set和episode信息: {video_path}")
                continue
            
            # 生成输出文件名
            output_filename = f"{set_name}_{episode_name}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"无法打开视频: {video_path}")
                continue
            
            # 获取总帧数
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                print(f"视频没有帧: {video_path}")
                cap.release()
                continue
            
            # 跳转到最后一帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - xx_last_frame)
            ret, frame = cap.read()
            
            if ret:
                # 保存最后一帧
                cv2.imwrite(output_path, frame)
                print(f"已保存:\n {output_path}")
            else:
                print(f"无法读取最后一帧: {video_path}")
            
            # 释放资源
            cap.release()
            
        except Exception as e:
            print(f"处理视频时出错 {video_path}: {str(e)}")

if __name__ == "__main__":
    # 指定要遍历的根目录
    root_directory = "/home/caijunhao/h-ceph/InternData-A1-raw/arx_lift2/Pick_the_industrial_components_from_the_conveyor"  # 当前目录，您可以修改为您的目录路径
    output_path = 'data/Pick_the_industrial_components_from_the_conveyor/'
    os.makedirs(output_path, exist_ok=True)
    sub_list = os.listdir(root_directory)
    exclude_list = []
    # exclude_list = [f"{i}" for i in range(16)] + [f"{i}" for i in range(26, 29)]
    xx_last_frame = 1
    # import pdb
    # pdb.set_trace()
    for sub in tqdm(sub_list):
        if sub.split('-')[1].split('_')[0] in exclude_list:
            continue
        # print("os.path.join([root_directory, sub])\n", os.path.join(root_directory, sub))
        extract_last_frame_from_videos(os.path.join(root_directory, sub), output_path, xx_last_frame=xx_last_frame)
    print("处理完成！")