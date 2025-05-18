import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def ave_compute(file):
    acc_matrix=np.loadtxt(file)
    ave=np.zeros(acc_matrix.shape[0],dtype=float)
    acc_d = -np.sort(-np.array(acc_matrix.ravel(),dtype=float))
    for i in range((acc_matrix.shape[0])):
        ave[i]=np.mean(acc_matrix[i, :i+1])
    nrows = len(acc_matrix)
    aver_total = np.mean(acc_d[:int(nrows*(nrows+1)/2)])
    return ave, aver_total, nrows

def ave_compute_fog(file):
	forget_matrix = np.loadtxt(file)
	ave = np.zeros(forget_matrix.shape[0]-1,dtype=float)
	for i in range((forget_matrix.shape[0])-1):
		ave[i] = np.mean(forget_matrix[i+1,:i+1])
	forget_vector = forget_matrix.ravel()
	forget_sorted = -np.sort(-np.array(forget_vector.ravel(),dtype=float))
	nrows = len(forget_matrix)
	forget_aver = np.mean(forget_sorted[:int(nrows*(nrows-1)/2)])
	return ave, forget_aver


target_tasks = '10'
folder_pattern = rf"cifar100_lucir-sr_{target_tasks}tasks_gamma1(\d+\.?\d*)"
current_dir = os.getcwd()
all_folders = os.listdir(current_dir)

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
        return f"γ1={gamma1}"
    return label.upper()

linestyles = [(0, (1, 4)), '--', '-.', ':', (0, (1, 2)), (0, (1, 3))]
markers = ['o', 's', 'd', 'v', '*', 'p']

acc_aver = []
forget_aver = []
taw_second_nums = []
taw_fifth_nums = []
taw_tenth_nums = []

tag_second_nums = []
tag_fifth_nums = []
tag_tenth_nums = []


# 收集数据
for fo in folders:
    # 收集acc_taw数据
    os.chdir('./{}/results'.format(fo))
    current_dir = os.getcwd()
    all_files = os.listdir(current_dir)
   
    # 收集平均准确率
    pattern = r"^acc_taw.*"
    files = [f for f in all_files if re.match(pattern, f)]
    files_n = sorted(files)
    temp, total_ave, length = ave_compute(files_n[-1])
    acc_aver.append(total_ave)
   
    # 收集遗忘率
    pattern = r"^forg_taw.*"
    files = [f for f in all_files if re.match(pattern, f)]
    files_n = sorted(files)
    temp, forget_ave = ave_compute_fog(files_n[-1])
    forget_aver.append(forget_ave)
   
    # 收集任务准确率
    pattern = r"^avg_accs_taw.*"
    files = [f for f in all_files if re.match(pattern, f)]
    files_n = sorted(files)
    if files_n:
        latest_file = files_n[-1]
        data = np.loadtxt(latest_file)
        if len(data) >= 10:  # 确保数据足够
            taw_second_nums.append(data[1])
            taw_fifth_nums.append(data[4])
            taw_tenth_nums.append(data[9])
        else:
            print(f"Warning: Not enough task data in {fo}")
            taw_second_nums.append(0)
            taw_fifth_nums.append(0)
            taw_tenth_nums.append(0)
   
    os.chdir('../../')

# 检查数据是否完整
print(f"Number of folders: {len(folders)}")
print(f"Accuracy data: {len(acc_aver)}")
print(f"Forgetting data: {len(forget_aver)}")
print(f"Task 2 data: {len(taw_second_nums)}")
print(f"Task 5 data: {len(taw_fifth_nums)}")
print(f"Task 10 data: {len(taw_tenth_nums)}")

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
    print("LUCIR-SR (γ1={:.1f}), average accuracy:{:.3f}".format(extract_lamb(fo), total_ave))
    os.chdir('../../')
    x_axis = range(1, length+1)
    plt.yticks(np.arange(0.55, 0.75 + 0.025, 0.025))
    plt.plot(x_axis, temp, linestyle=linestyles[i], marker=markers[i], label=clean_label(fo))
    i += 1
    plt.title('Average Accuracy (LUCIR-SR)')
    plt.legend(loc='lower left', ncol=2, fontsize='small')
    plt.savefig('acc_'+folder_pattern+'.eps',dpi=800,format='eps')
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
    plt.savefig('forget_'+folder_pattern+'.eps',dpi=800,format='eps')


# 打印结果表格
print("\n=== Results Summary ===")
print("γ1 Values:", end="")
for fo in folders:
    lamb = extract_lamb(fo)
    print(f"&{gamma1:.1f}", end="")
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
        print(f"γ1={lamb:.1f}: Task 2={data[1]:.3f}, Task 5={data[4]:.3f}, Task 10={data[9]:.3f}")
    os.chdir('../../')


# 输出详细结果
print("\n=== 详细结果 ===")
# 输出文件夹名和lambda值
print("文件夹名称:", end="")
for fo in folders:
    print(f"&{fo}", end="")
print("\\\\")

# 输出平均准确率
print("平均准确率:", end="")
for acc in acc_aver:
    print(f"&{acc:.3f}", end="")
print("\\\\")

# 输出平均遗忘率
print("平均遗忘率:", end="")
for forg in forget_aver:
    print(f"&{forg:.3f}", end="")
print("\\\\")

# 处理并输出TAW数据
print("\n=== TAW数据 ===")
for fo in folders:
    os.chdir('./{}/results'.format(fo))
    current_dir = os.getcwd()
    all_files = os.listdir(current_dir)
    pattern = r"^avg_accs_taw.*"
    files = [f for f in all_files if re.match(pattern, f)]
    files_n = sorted(files)
    if files_n:
        latest_file = files_n[-1]
        data = np.loadtxt(latest_file)
        taw_second_nums.append(data[1])
        taw_fifth_nums.append(data[4])
        taw_tenth_nums.append(data[9])
    os.chdir('../../')

print("TAW Task 2:", end="")
for num in taw_second_nums:
    print(f"&{num:.3f}", end="")
print("\\\\")

print("TAW Task 5:", end="")
for num in taw_fifth_nums:
    print(f"&{num:.3f}", end="")
print("\\\\")

print("TAW Task 10:", end="")
for num in taw_tenth_nums:
    print(f"&{num:.3f}", end="")
print("\\\\")

# 处理并输出TAG数据
print("\n=== TAG数据 ===")
for fo in folders:
    os.chdir('./{}/results'.format(fo))
    current_dir = os.getcwd()
    all_files = os.listdir(current_dir)
    pattern = r"^avg_accs_tag.*"
    files = [f for f in all_files if re.match(pattern, f)]
    files_n = sorted(files)
    if files_n:
        latest_file = files_n[-1]
        data = np.loadtxt(latest_file)
        tag_second_nums.append(data[1])
        tag_fifth_nums.append(data[4])
        tag_tenth_nums.append(data[9])
    os.chdir('../../')

print("TAG Task 2:", end="")
for num in tag_second_nums:
    print(f"&{num:.3f}", end="")
print("\\\\")

print("TAG Task 5:", end="")
for num in tag_fifth_nums:
    print(f"&{num:.3f}", end="")
print("\\\\")

print("TAG Task 10:", end="")
for num in tag_tenth_nums:
    print(f"&{num:.3f}", end="")
print("\\\\")