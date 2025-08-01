import matplotlib.pyplot as plt
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#plt.rcParams['font.serif'] = "Times New Roman"
#plt.rcParams['font.family'] = "serif"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

c1 = (31.0/255, 119/255.0, 180/255.0)
c2 = (174/255.0, 199/255.0, 232/255.0)
c3 = (1, 127/255.0, 14/255.0)
c4 = (1, 187/255.0, 120/255.0)
c5 = (44/255.0, 160/255.0, 44/255.0)
c6 = (152/255.0, 223/255.0, 138/255.0)
c7 = (214/255.0, 39/255.0, 40/255.0)
c8 = (1, 152/255.0, 150/255.0)
c9 = (148/255.0, 103/255.0, 189/255.0)
c10 = (192/255.0, 176/255.0, 213/255.0)


act_hf = np.array([84557168640, 84557168640, 97978941440]) / 1e9
act_prune = np.array([21474836480, 21474836480, 28185722880]) / 1e9
act_prune_remat = np.array([16106127360, 21474836480, 20132659200]) / 1e9
act_prune_remat_tok = np.array([12499025920, 12834570240, 12750684160]) / 1e9

fig = plt.figure(figsize=(12, 7))
ax1 = fig.add_subplot(111)

# Data for the bar chart
categories = ['LoRA', 'Adapter', 'IA3']

# Set up the bar positions
bar_width = 0.2
index = np.arange(len(categories))
space_width = 0.01

# Create the bar chart
plt.bar(index, act_prune_remat_tok, width=bar_width, label='Collie', color=c1)

index = index + bar_width + space_width
plt.bar(index, act_prune_remat, width=bar_width, label='Collie w/o Token-Level Finetuning', color=c2)

index = index + bar_width + space_width
plt.bar(index, act_prune, width=bar_width, label='Collie w/o Token-Level Finetuning + Rematerization', color=c3)

index = index + bar_width + space_width
plt.bar(index, act_hf, width=bar_width, label='Collie w/o Token-Level Finetuning + Rematerization + Graph Pruning', color=c4)


# Customize the chart
plt.xlabel('Finetuning methods', fontsize=11, fontweight='bold')
plt.ylabel('Activation Memory Requirement (GB)', fontsize=11, fontweight='bold')
plt.xticks(index - 1.5 * bar_width, categories)
plt.ylim(0, 140)
plt.legend(fontsize=12)
# Show the chart
# plt.show()
output_folder = "./output"
os.makedirs(output_folder, exist_ok=True)
plt.savefig(os.path.join(output_folder, "fig13.pdf"), dpi=300, bbox_inches='tight')