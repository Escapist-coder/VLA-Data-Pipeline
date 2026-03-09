import os
import h5py
import glob
import concurrent.futures
import multiprocessing
from functools import partial
from tqdm import tqdm

# 导入之前写的模块
from bridge_parser import VLADatasetParser
from kinematic_filter import KinematicFilter
from time_sync import TimeSynchronizer
from vlm_relabel import VLMRelabeler

def process_single_trajectory(file_path, output_folder):
    """
    负责在独立进程中处理单个文件的“打工函数”。
    【面试高频点】：为什么要把组件的实例化放在函数内部？
    答：为了防止多进程间的内存冲突和序列化报错(PicklingError)。让每个进程拥有自己独立的清洗器和解析器。
    """
    file_name = os.path.basename(file_path)
    
    # 实例化四大组件 (每个 CPU 核心都会拥有自己独立的一套工具)
    filter_engine = KinematicFilter(min_length=15)
    sync_engine = TimeSynchronizer(target_fps=30)
    vlm_engine = VLMRelabeler() 

    # ==========================================
    # 步骤 1：解析提取
    # ==========================================
    parser = VLADatasetParser(file_path)
    parser.extract_trajectory()
    data = parser.data_dict
    
    if 'images' not in data or 'ee_pose' not in data or 'actions' not in data:
        return False # 解析失败，向主进程汇报 False
        
    # ==========================================
    # 步骤 2：物理规则质检
    # ==========================================
    passed, reasons, _, _ = filter_engine.run_checks(data['ee_pose'], data['actions'])
    if not passed:
        return False # 质检被拦截，汇报 False
        
    # ==========================================
    # 步骤 3：多模态时间戳对齐
    # ==========================================
    synced_imgs, synced_qpos, synced_actions = sync_engine.synchronize(
        data['images'], data['ee_pose'], data['actions']
    )
    
    # ==========================================
    # 步骤 4：VLM 自动化重打标
    # ==========================================
    from PIL import Image
    img_start = Image.fromarray(synced_imgs[0])
    img_end = Image.fromarray(synced_imgs[-1])
    new_instruction = vlm_engine.generate_rich_instruction(img_start, img_end)

    # ==========================================
    # 步骤 5：落盘存储
    # ==========================================
    out_path = os.path.join(output_folder, f"clean_{file_name}")
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('images', data=synced_imgs)
        f.create_dataset('qpos', data=synced_qpos)
        f.create_dataset('actions', data=synced_actions)
        f.create_dataset('instruction', data=new_instruction)
        
    return True # 成功走完全程，向主进程汇报 True


def run_parallel_pipeline(input_folder, output_folder):
    print("🚀 启动具身智能大规模数据清洗管线 (多进程提速版) 🚀\n")
    os.makedirs(output_folder, exist_ok=True)
    
    all_files = glob.glob(os.path.join(input_folder, "*.hdf5"))
    print(f"📂 扫描到 {len(all_files)} 个待处理的原始轨迹文件。\n")

    if not all_files:
        print("未找到任何文件，退出程序。")
        return

    # 自动获取你电脑的 CPU 核心数
    max_cores = multiprocessing.cpu_count()
    print(f"⚡ 启动进程池！检测到 {max_cores} 个 CPU 核心，火力全开...")

    # 使用 partial 固定 output_folder 参数，这样 process_func 就变成了一个只需要传 file_path 的函数
    process_func = partial(process_single_trajectory, output_folder=output_folder)

    # 核心：启动进程池并发执行
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
        # executor.map 会把 all_files 里的文件自动分配给空闲的 CPU 核心
        # tqdm 用来监控这个并行任务的进度条，list() 会收集所有的 True/False 结果
        results = list(tqdm(executor.map(process_func, all_files), total=len(all_files), desc="Parallel Processing"))

    # 因为 True 的数值是 1，False 是 0，直接 sum() 就能算出成功了多少个文件
    valid_count = sum(results)

    print("\n" + "="*50)
    print("📊 多进程数据清洗任务完成报告")
    print(f"总计输入文件数: {len(all_files)}")
    print(f"成功清洗并对齐保留的文件数: {valid_count}")
    print(f"数据清洗率 (留存率): {(valid_count/len(all_files))*100:.1f}%")
    print(f"干净的数据已存入: {os.path.abspath(output_folder)}")
    print("="*50)

if __name__ == "__main__":
    run_parallel_pipeline(input_folder="test_data", output_folder="clean_data")