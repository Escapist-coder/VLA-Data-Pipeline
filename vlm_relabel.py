import h5py
import numpy as np
from PIL import Image
import time

class VLMRelabeler:
    def __init__(self, api_key=None):
        self.api_key = api_key
        # 如果你有真实的 API Key，这里会初始化真正的 VLM 客户端
        if self.api_key:
            print("🔗 已连接到真实的 VLM API...")
            # import google.generativeai as genai
            # genai.configure(api_key=self.api_key)
            # self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            print("⚠️ 未检测到 API Key，启用本地离线模拟 VLM 模式...")

    def extract_start_end_frames(self, h5_path):
        """从 HDF5 中提取轨迹的第一帧(初始状态)和最后一帧(结束状态)"""
        print(f"📦 正在打开数据文件: {h5_path}")
        try:
            with h5py.File(h5_path, 'r') as f:
                # 兼容我们之前生成的 test_data 结构
                images = f['observations']['images/top'][:]
                
                # 提取首尾两帧
                start_frame_np = images[0]
                end_frame_np = images[-1]
                
                # 将 numpy 数组转换为 PIL Image 格式 (VLM 接口通常需要这种格式)
                start_image = Image.fromarray(start_frame_np)
                end_image = Image.fromarray(end_frame_np)
                
                return start_image, end_image
        except Exception as e:
            print(f"❌ 提取图像失败: {e}")
            return None, None

    def generate_rich_instruction(self, start_img, end_img):
        """调用 VLM 生成高质量的动作描述文本"""
        prompt = """
        You are an expert roboticist. I will provide you with the start and end camera frames of a robot manipulation task.
        Please describe the task the robot just completed in a single, highly detailed sentence. 
        Include the objects involved, their colors, and the spatial relationship changes.
        """
        
        if self.api_key:
            # === 这里是真实的 API 调用逻辑 (留给你以后用) ===
            # response = self.model.generate_content([prompt, start_img, end_img])
            # return response.text.strip()
            pass
        else:
            # === 这里是模拟 VLM 思考和输出的过程 ===
            print("🧠 [模拟 VLM] 正在分析视觉差异特征...")
            time.sleep(1.5) # 模拟网络延迟
            
            # 我们随机返回一些高质量的伪造指令，展示重打标的效果
            mock_instructions = [
                "The robot arm successfully grasped the transparent plastic bottle and placed it upright inside the blue storage bin.",
                "The robotic manipulator picked up the red apple from the wooden table and gently dropped it into the metallic sink.",
                "The end-effector smoothly slid the yellow sponge across the white plate to clean its surface."
            ]
            import random
            return random.choice(mock_instructions)

if __name__ == "__main__":
    # 我们用生成的第一个完美测试文件来做实验
    target_file = "test_data/case1_perfect.hdf5"
    
    # 实例化打标器 (如果不传 api_key，就会走本地模拟逻辑，确保代码必能跑通)
    relabeler = VLMRelabeler(api_key=None) 
    
    print("\n" + "="*50)
    print("🎥 步骤 1: 提取首尾关键帧")
    img_start, img_end = relabeler.extract_start_end_frames(target_file)
    
    if img_start and img_end:
        print("\n✍️  步骤 2: 将多模态图像送入大模型请求重打标")
        new_instruction = relabeler.generate_rich_instruction(img_start, img_end)
        
        print("\n✅ VLM 重打标完成！")
        print(f"👉 生成的高质量指令: \n\033[92m\"{new_instruction}\"\033[0m")
        print("="*50)