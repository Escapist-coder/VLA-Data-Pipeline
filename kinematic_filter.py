import h5py
import numpy as np
import glob

class KinematicFilter:
    def __init__(self, min_length=15, movement_threshold=0.01, velocity_limit=1.0):
        # 规则 1：最短有效帧数 (太短的通常是误触录制)
        self.min_length = min_length
        # 规则 2：首尾位置最小位移 (判断是否在原地发呆，未完成任务)
        self.movement_threshold = movement_threshold
        # 规则 3：相邻两帧最大允许的关节角速度 (判断传感器是否发生数值爆炸/突变)
        self.velocity_limit = velocity_limit

    def run_checks(self, qpos, actions):
        """
        运行三大物理规则质检
        qpos: 本体状态数组 [帧数, 关节数]
        actions: 动作数组 [帧数, 动作维度]
        返回: (是否通过, 失败原因列表, 总位移量, 最大角速度)
        """
        num_frames = qpos.shape[0]
        reasons = []

        # 1. 轨迹过短剔除
        if num_frames < self.min_length:
            reasons.append(f"轨迹过短: 仅 {num_frames} 帧 (要求 >= {self.min_length})")

        # 2. 静止发呆剔除
        # 计算首帧和尾帧关节角度的 L2 距离 (欧氏距离)，衡量整段轨迹的变化量
        total_movement = np.linalg.norm(qpos[-1] - qpos[0])
        if total_movement < self.movement_threshold:
            reasons.append(f"轨迹疑似静止: 总位移量 {total_movement:.4f} < {self.movement_threshold}")

        # 3. 奇异点/突变剔除 (面试高频考点！)
        # np.diff 计算相邻两帧的差值，这就代表了瞬时速度
        velocities = np.diff(qpos, axis=0) 
        max_velocity = np.max(np.abs(velocities))
        if max_velocity > self.velocity_limit:
            reasons.append(f"发现速度突变: 最大瞬时角速度 {max_velocity:.4f} > 阈值 {self.velocity_limit}")

        passed = len(reasons) == 0
        return passed, reasons, total_movement, max_velocity

def process_single_file(h5_path):
    print(f"🛡️ 开始质检文件: {h5_path}")
    try:
        with h5py.File(h5_path, 'r') as f:
            # 智能路由提取数据 (兼容昨天写的两种格式)
            if 'action' in f and 'observations' in f:
                actions = f['action'][:]
                qpos = f['observations']['qpos'][:]
            elif 'data' in f:
                demo = f['data'][list(f['data'].keys())[0]]
                actions = demo['actions'][:]
                qpos = demo['obs']['joint_positions'][:]
            else:
                print("❌ 数据集格式无法识别！")
                return
            
            # 初始化过滤器并运行质检
            # 这里的阈值(0.01和1.0)是经验值，大厂里通常会根据具体机械臂的物理极限来设置
            data_filter = KinematicFilter(min_length=15, movement_threshold=0.01, velocity_limit=1.0)
            passed, reasons, movement, max_vel = data_filter.run_checks(qpos, actions)

            print("\n=== 📝 质检报告 ===")
            print(f"轨迹长度: {qpos.shape[0]} 帧")
            print(f"总位移量: {movement:.4f}")
            print(f"最大角速度: {max_vel:.4f}")
            
            if passed:
                print("✅ 结论: 质检通过！这是一条高质量的有效轨迹。")
            else:
                print("❌ 结论: 质检失败！该轨迹将被丢弃。")
                for r in reasons:
                    print(f"   - 原因: {r}")

    except Exception as e:
        print(f"读取文件时发生错误: {e}")

if __name__ == "__main__":
    # 批量寻找 test_data 文件夹下的所有 hdf5 文件
    test_files = glob.glob("test_data/*.hdf5")
    
    print("🚀 启动数据清洗管线批量质检...\n")
    for file_path in sorted(test_files):
        process_single_file(file_path)
        print("-" * 50)