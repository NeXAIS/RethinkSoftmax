 import re
import os
import numpy as np
import matplotlib.pyplot as plt

# ... 保留ave_compute和ave_compute_fog函数 ...

# 修改folder_pattern只匹配lucir-sr的结果
target_tasks = '10'
folder_pattern = rf"cifar100_lucir-sr_{target_tasks}tasks_lamb(\d+\.?\d*)"
current_dir = os.getcwd()
all_folders = os.listdir(current_dir)

# 获取所有符合条件的文件夹并按lamb值排序
def extract_lamb(folder_name):
    match = re.match(folder_pattern, folder_name)
    if match:
        return float(match.group(1))
    return float('inf')

folders = [fo for fo in all_folders if re.match(folder_pattern, fo)]
folders = sorted(folders, key=extract_lamb)

def clean_label(label):
    match = re.match(folder_pattern, label)
    if match:
        lamb = float(match.group(1))
        return f"λ={lamb}"
    return label.upper()

# 设置绘图样式
linestyles = [(0, (1, 4)), '--', '-.', ':', (0, (1, 2)), (0, (1, 3))]
markers = ['o', 's', 'd', 'v', '*', 'p']

acc_aver = []
forget_aver = []

# 绘制平均准确率图
plt.figure(1)
i = 0
for fo in folders:
    os.chdir('./{}/results'.format(fo))
    current_dir = os.getcwd()
    all_files = os.listdir(current_dir)
    pattern = r"^acc_taw.*"
    
    files = [f for f in all_files if re.match(pattern, f)]
    files_n = sorted(files)
    print(files_n[-1])
    temp, total_ave, length = ave_compute(files_n[-1])
    print("LUCIR-SR (λ={:.1f}), average accuracy:{:.3f}".format(extract_lamb(fo), total_ave))
    
    os.chdir('../../')
    x_axis = range(1, length+1)
    plt.yticks(np.arange(0.55, 0.75 + 0.025, 0.025))
    plt.plot(x_axis, temp, linestyle=linestyles[i], marker=markers[i], label=clean_label(fo))
    i += 1
    plt.title('Average Accuracy (LUCIR-SR)')
    plt.legend(loc='lower left', ncol=2, fontsize='small')
    plt.savefig('acc_lucir-sr_comparison.eps', dpi=800, format='eps')
    acc_aver.append(total_ave)

# 绘制遗忘率图
plt.figure(2)
i = 0
for fo in folders:
    os.chdir('./{}/results'.format(fo))
    current_dir = os.getcwd()
    all_files = os.listdir(current_dir)
    pattern = r"^forg_taw.*"
    files = [f for f in all_files if re.match(pattern, f)]
    files_n = sorted(files)
    temp, forget_ave = ave_compute_fog(files_n[-1])
    forget_aver.append(forget_ave)
    print("LUCIR-SR (λ={:.1f}), average forgetting:{:.3f}".format(extract_lamb(fo), forget_ave))

    os.chdir('../../')
    x_axis = range(2, length+1)
    plt.yticks(np.arange(0, 1, 0.1))
    plt.plot(x_axis, temp, linestyle=linestyles[i], marker=markers[i], label=clean_label(fo))
    plt.title('Average Forgetting (LUCIR-SR)')
    plt.legend(loc='upper left', ncol=2, fontsize='small')
    i += 1
    plt.savefig('forget_lucir-sr_comparison.eps', dpi=800, format='eps')

# 打印结果表格
print("\n=== Results Summary ===")
print("Lambda Values:", end="")
for fo in folders:
    lamb = extract_lamb(fo)
    print(f"&{lamb:.1f}", end="")
print("\\\\")

print("Avg Accuracy:", end="")
for acc in acc_aver:
    print(f"&{acc:.3f}", end="")
print("\\\\")

print("Avg Forgetting:", end="")
for forg in forget_aver:
    print(f"&{forg:.3f}", end="")
print("\\\\")

# 处理avg_accs_taw数据
print("\n=== Task-wise Accuracy ===")
for fo in folders:
    lamb = extract_lamb(fo)
    os.chdir('./{}/results'.format(fo))
    current_dir = os.getcwd()
    all_files = os.listdir(current_dir)
    pattern = r"^avg_accs_taw.*"
    files = [f for f in all_files if re.match(pattern, f)]
    files_n = sorted(files)
    if files_n:
        latest_file = files_n[-1]
        data = np.loadtxt(latest_file)
        print(f"λ={lamb:.1f}: Task 2={data[1]:.3f}, Task 5={data[4]:.3f}, Task 10={data[9]:.3f}")
    os.chdir('../../')