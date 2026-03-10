# 🤖 VLA-Data-Pipeline: Embodied AI Data Auto-Cleaning & Relabeling

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data Format](https://img.shields.io/badge/Data-HDF5%20%7C%20Zarr-orange.svg)]()

> 面向具身智能与 VLA 模型的大规模多模态轨迹数据**自动化清洗、对齐与重打标**工程管线。

---

## 🎥 效果演示 (Demo)

<div align="center">
  

https://github.com/user-attachments/assets/02f7dc55-9ec4-4891-a823-2db0282b5437


  <p><i>示意视频: 经过 Pipeline 清洗并由 VLM 自动生成高质量指令的具身操作轨迹</i></p>
</div>

本管线对原始数据的核心提升，直观体现在**多模态对齐**与**语义丰富度**上，下图为描述性的对比表格：

| 📉 处理前 (Raw Dirty Data) | 📈 本管线处理后 (Cleaned & Relabeled) |
| :--- | :--- |
| ❌ **物理状态:** 包含未对齐的 50Hz 高频噪声 | ✅ **物理状态:** 30Hz 多模态时间戳完美对齐 |
| ❌ **异常数据:** 夹杂短尾断裂、速度突变帧 | ✅ **异常数据:** 经 Kinematic Filter 拦截剔除 |
| ❌ **原始标签:** `"chair truncated"` | ✅ **VLM 丰富标签:** `"The robot pushes the chair under the table to organize the spatial arrangement."` |

*(注：左侧模拟原始未经清洗的粗糙状态，右侧为经过 Pipeline 标准化落盘后的高质量训练集状态。)*

## 📌 项目背景 (Background)
在人形机器人与双臂灵巧操作的 VLA 模型训练中，**“数据决定了模型的上限”**。真实遥操作采集的数据往往充满噪声。本项目致力于解决以下三大行业痛点：
1. **异构传感器频率未对齐**（如 30Hz 视觉与 50Hz 本体状态错位）。
2. **运动学异常与噪声**（如传感器丢包导致的末端速度突变、静止发呆轨迹）。
3. **人类文本指令匮乏**（粗粒度标签导致模型语言泛化性极差）。

## ⚙️ 系统架构 (System Architecture)

本项目包含四大核心引擎，构成完整的串行数据处理 DAG（有向无环图）：

1. `Parser Engine`: 智能路由 HDF5 解析器，兼容 `ALOHA` 与 `BridgeData V2` 异构树状结构。
2. `Kinematic Filter`: 运动学质检器，基于 $L_2$ 范数与一阶差分 $\Delta q / \Delta t$ 拦截物理违规轨迹。
3. `Sync Engine`: 多模态对齐器，采用连续变量 `Linear` + 离散动作 `Nearest-neighbor` 的混合插值策略。
4. `VLM Relabeler`: 视觉大模型引擎，提取轨迹首尾关键帧 ($I_0, I_T$)，可调用 Gemini 1.5 Flash 接口生成细粒度空间语义标签。

## 🚀 性能基准测试 (Performance Benchmark)

得益于 `concurrent.futures.ProcessPoolExecutor` 引入的**多进程并发架构 (Multiprocessing)**，本管线在处理 IO+CPU 密集型任务时表现优异，并行效率达约80%：

| 环境配置 | 数据规模 | 原始预计耗时 (单核) | 本管线耗时 (多进程) | 加速比 |
| :--- | :--- | :--- | :--- | :--- |
| 8-Core CPU, NVMe SSD | 50 GB (~500 HDF5) | ~ 65 分钟 | **~ 12 分钟** | **5.4x** |


## 🛠️ 快速复现 (Quick Start)

本项目内置了微型测试数据，您可以零成本一键跑通整个流水线，请参照[项目博客](https://blog.csdn.net/2303_77547168/article/details/158650625?sharetype=blogdetail&sharerId=158650625&sharerefer=PC&sharesource=2303_77547168&spm=1011.2480.3001.8118)进行操作后运行主干流水线 (使用 example_data 里的微型数据)


```bash
python main_pipeline.py --input example_data --output clean_data
```

另外，在实际操作时的方法与重打标可视化等也在博客中，如有兴趣请详细阅读
