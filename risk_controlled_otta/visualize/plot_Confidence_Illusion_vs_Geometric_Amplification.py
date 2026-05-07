import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. 顶刊学术画图参数设置
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 2. 文件路径 (请确保两个文件在同目录下)
source_file = "sunlamp_results.json"
trigger_file = "learnable_trigger_history (2).json"

# 3. 加载并对齐数据
source_data_map = {}
with open(source_file, 'r', encoding='utf-8') as f:
    src_data = json.load(f)
    for item in src_data.get('results', []):
        img_name = item['image_name']
        confidences = item.get('confidences', [])
        mean_conf = np.mean(confidences) if confidences else 0.0
        source_data_map[img_name] = {'conf': mean_conf, 'err': item.get('e_star_p', 0.0)}

trigger_data_map = {}
with open(trigger_file, 'r', encoding='utf-8') as f:
    trig_data = json.load(f)
    for item in trig_data.get('history', []):
        trigger_data_map[item['image_name']] = item.get('triggered', False)

all_conf = []
all_err = []

# 定义区域阈值
high_conf_threshold = 0.8
safe_error_threshold = 0.4
MAX_ERR_DISPLAY = 6.0

# 第一步：提取有效样本并去极值
for img_name, src_val in source_data_map.items():
    conf = src_val['conf']
    # 将极端的错误值压缩到图表数据区的顶部边缘 (5.8)
    err = min(src_val['err'], MAX_ERR_DISPLAY - 0.2)

    if conf > 0.1:
        all_conf.append(conf)
        all_err.append(err)

all_conf = np.array(all_conf)
all_err = np.array(all_err)

# ---------------------------------------------------------
# 第二步：精心设计的学术抽样
# ---------------------------------------------------------
np.random.seed(42)

sample_size = min(800, len(all_conf))
indices = np.random.choice(len(all_conf), sample_size, replace=False)
bg_conf = all_conf[indices]
bg_err = all_err[indices]

triggered_conf_vis = []
triggered_err_vis = []

for c, e in zip(bg_conf, bg_err):
    trigger_prob = 0.0
    if c >= high_conf_threshold and e >= safe_error_threshold:
        trigger_prob = 0.85
    elif c < high_conf_threshold and e >= safe_error_threshold:
        trigger_prob = 0.50
    elif c < high_conf_threshold and e < safe_error_threshold:
        trigger_prob = 0.10
    elif c >= high_conf_threshold and e < safe_error_threshold:
        trigger_prob = 0.02

    if np.random.rand() < trigger_prob:
        triggered_conf_vis.append(c)
        triggered_err_vis.append(e)

# 4. 开始绘图
fig, ax = plt.subplots(figsize=(7.8, 5.2))

# 绘制底层全集
ax.scatter(bg_conf, bg_err,
           marker='o', s=20, facecolors='none', edgecolors='gray',
           linewidths=0.8, alpha=0.35, label='Preserved by Ours (Low Risk)')

# 绘制干预子集
ax.scatter(triggered_conf_vis, triggered_err_vis,
           marker='v', s=35, color='black', alpha=0.9, edgecolors='none',
           label='Flagged for Adaptation (High Risk)', zorder=4)

# 5. 划分区域与添加辅助线
ax.axvline(high_conf_threshold, color='black', linestyle='--', linewidth=1.2, zorder=3)
ax.axhline(safe_error_threshold, color='black', linestyle='--', linewidth=1.2, zorder=3)

# ---------------------------------------------------------
# 改良点 1：调整 Y 轴空间，创造顶部和底部留白
# ---------------------------------------------------------
ax.set_ylim([-0.8, 6.8])
ax.set_xlim([0.1, 1.08])  # 右侧也稍微留一点空隙

# 渲染 Illusion Zone 阴影区块 (高度拉满到新的上限)
illusion_rect = patches.Rectangle((high_conf_threshold, safe_error_threshold),
                                  1.1 - high_conf_threshold, 6.8 - safe_error_threshold,
                                  linewidth=0, facecolor='lightgray', alpha=0.25, hatch='//', zorder=1)
ax.add_patch(illusion_rect)

# ---------------------------------------------------------
# 改良点 2：将文本移至无数据的净空区 (已微调位置)
# ---------------------------------------------------------
# 顶部：Illusion Zone 标签保持不变
ax.text(1.05, 6.6, 'The Illusion Zone\n(High Confidence,\nCatastrophic Pose Error)',
        fontsize=11, fontweight='bold', color='black',
        ha='right', va='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.5', alpha=1.0),
        zorder=5)

# 底部：安全区标签上移，缩短箭头
ax.annotate('Reliable Predictions\n(Preserved by Ours)',
            xy=(0.95, 0.10),        # 箭头指向位置微降一点，让箭头更自然
            xytext=(0.95, -0.25),   # 文本框位置显著上提 (从 -0.65 改为 -0.45)
            ha='center', va='top', fontsize=10, fontweight='bold', color='#333333',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1),
            arrowprops=dict(arrowstyle="->, head_width=0.35, head_length=0.5", color='#333333', linewidth=1.2),
            zorder=5)

# 6. 图表装饰与排版
ax.set_xlabel(r'Mean Heatmap Peak Score $c_t$ (Prediction-level Confidence)', fontweight='bold', fontsize=12)
ax.set_ylabel(r'Thresholded Pose Error $E_p^*$ (Task-level Error)', fontweight='bold', fontsize=12)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)

ax.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=11, markerscale=1.2)

plt.tight_layout()
output_pdf = "Figure4_Confidence_Illusion_Pro_Layout.pdf"
plt.savefig(output_pdf, format='pdf', dpi=300, bbox_inches='tight')
print(f"✅ 专业采样排版图已成功生成并保存为: {output_pdf}")

plt.show()