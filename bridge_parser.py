import h5py
import numpy as np
import cv2
import os

class VLADatasetParser:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.data_dict = {}
        
    def extract_trajectory(self):
        """
        核心能力：智能搜索并提取 HDF5 中的核心字段
        支持格式：Robomimic / Bridge V2 / ALOHA / 我们的测试用例
        """
        print(f"🔄 正在解析数据集: {self.h5_path}")
        try:
            with h5py.File(self.h5_path, 'r') as f:
                # ==========================================
                # 智能路由：判断根目录结构
                # ==========================================
                if 'data' in f:
                    demo_key = list(f['data'].keys())[0] # 找到类似 demo_0 的组
                    demo = f['data'][demo_key]
                else:
                    demo = f # ALOHA 格式，直接在根目录

                # ==========================================
                # 1. 提取图像 (加入对多种键名的兼容)
                # ==========================================
                if 'obs' in demo and 'agentview_image' in demo['obs']:
                    self.data_dict['images'] = demo['obs']['agentview_image'][:]
                elif 'observations' in demo and 'images0' in demo['observations']:
                    self.data_dict['images'] = demo['observations']['images0'][:]
                # 兼容我们生成的测试数据: observations/images/top
                elif 'observations' in demo and 'images' in demo['observations'] and 'top' in demo['observations']['images']:
                    self.data_dict['images'] = demo['observations']['images']['top'][:]
                else:
                    raise ValueError("未找到标准图像字段！")

                # ==========================================
                # 2. 提取本体状态 (ee_pose 或 qpos)
                # ==========================================
                if 'obs' in demo and 'robot0_eef_pos' in demo['obs']:
                    self.data_dict['ee_pose'] = demo['obs']['robot0_eef_pos'][:]
                elif 'obs' in demo and 'joint_positions' in demo['obs']:
                    self.data_dict['ee_pose'] = demo['obs']['joint_positions'][:]
                # 兼容 ALOHA 测试数据: observations/qpos
                elif 'observations' in demo and 'qpos' in demo['observations']:
                    self.data_dict['ee_pose'] = demo['observations']['qpos'][:]
                else:
                    raise ValueError("未找到机械臂状态字段！")

                # ==========================================
                # 3. 提取动作 (actions 或 action)
                # ==========================================
                if 'actions' in demo:
                    self.data_dict['actions'] = demo['actions'][:]
                # 兼容 ALOHA 测试数据: action
                elif 'action' in demo:
                    self.data_dict['actions'] = demo['action'][:]
                else:
                    raise ValueError("未找到动作字段！")

                # ==========================================
                # 4. 提取语言指令 (如果存在)
                # ==========================================
                if 'language_instruction' in demo:
                    raw_text = demo['language_instruction'][()]
                    self.data_dict['instruction'] = raw_text.decode('utf-8') if isinstance(raw_text, bytes) else str(raw_text)
                else:
                    self.data_dict['instruction'] = "No instruction provided."
                    
        except Exception as e:
            print(f"❌ 解析失败: {e}")

    def export_to_mp4(self, output_filename="trajectory_output.mp4", fps=30):
        """
        将提取到的图像序列合成为可视化的 MP4 视频
        对应 Day 1 目标：把轨迹保存为 .mp4 视频用来肉眼观察
        """
        if 'images' not in self.data_dict:
            print("❌ 字典中没有图像数据，无法生成视频！")
            return

        images = self.data_dict['images']
        num_frames, height, width, channels = images.shape
        
        print(f"\n🎬 正在生成视频: {output_filename} ({num_frames} 帧)...")
        
        # 定义视频编码器 (mp4v 是广泛支持的 mp4 编码)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        for i in range(num_frames):
            frame = images[i]
            # OpenCV 默认使用 BGR 色彩空间，如果原图是 RGB，需要转换
            # 真实具身数据通常是 RGB，这里做一次容错转换
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        out.release()
        print(f"✅ 视频生成完毕！已保存至: {os.path.abspath(output_filename)}")


if __name__ == "__main__":
    # 使用我们刚刚的通用解析器
    target_file = "sample_dataset.hdf5" # 稍后我们会换成真正的 Bridge 数据
    
    parser = VLADatasetParser(target_file)
    parser.extract_trajectory()
    parser.export_to_mp4("day1_result.mp4")