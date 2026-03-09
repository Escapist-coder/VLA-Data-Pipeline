import h5py
import os

def inspect_clean_data(file_path):
    print(f"🕵️ 开始开箱检验成品数据: {file_path}")
    if not os.path.exists(file_path):
        print("❌ 找不到文件！请确认你的 main_pipeline 成功生成了 clean_data 目录。")
        return

    with h5py.File(file_path, 'r') as f:
        print("\n=== 📦 数据集内部结构 ===")
        print(f"📷 图像 (images): {f['images'].shape}, 类型: {f['images'].dtype}")
        print(f"🦾 关节 (qpos):   {f['qpos'].shape}, 类型: {f['qpos'].dtype}")
        print(f"⚡ 动作 (actions): {f['actions'].shape}, 类型: {f['actions'].dtype}")
        
        # 提取并解码 VLM 生成的标签
        instruction = f['instruction'][()]
        if isinstance(instruction, bytes):
            instruction = instruction.decode('utf-8')
            
        print("\n=== 🏷️ AI 自动生成的高质量语义标签 ===")
        print(f"\033[92m\"{instruction}\"\033[0m\n")

if __name__ == "__main__":
    # 指向我们刚清洗出来的成品文件
    target_file = "clean_data/clean_case1_perfect.hdf5"
    inspect_clean_data(target_file)