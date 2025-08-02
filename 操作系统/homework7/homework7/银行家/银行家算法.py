import random
import time
import copy

def bankers_algorithm(x, resources, task, Max, Allocation, Need, Available, req_num):
    """
    实现银行家算法，检查请求是否安全可分配。

    参数:
        x (int): 任务数量。
        resources (list): 请求的三个资源数量 [a, b, c]。
        task (int): 请求的任务编号。
        req_num (int): 请求的编号。
        Max (list): Max 矩阵。
        Allocation (list): Allocation 矩阵。
        Need (list): Need 矩阵。
        Available (list): Available 资源列表。
        Osmax (list): 系统资源上限列表。

    返回:
        tuple: (x, task, req_num, Max, Allocation, Need, Available, flag)
            flag (bool): 请求是否被成功分配。
    """
    # 步骤 1: 检查请求是否超过任务的最大需求
    for i in range(3):
        if resources[i] > Need[task][i]:
            print(f"请求 {req_num} 被拒绝：请求的资源超过任务 {task} 的需求。")
            return (x, task, Max, Allocation, Need, Available, False)

    # 步骤 2: 检查系统是否有足够的资源可用
    for i in range(3):
        if resources[i] > Available[i]:
            print(f"请求 {req_num} 被拒绝：系统资源不足以满足任务 {task} 的请求。")
            return (x, task, Max, Allocation, Need, Available, False)

    # 步骤 3: 创建资源分配的副本，尝试分配资源
    try_Max = copy.deepcopy(Max)
    try_Allocation = copy.deepcopy(Allocation)
    try_Need = copy.deepcopy(Need)
    try_Available = copy.deepcopy(Available)

    # 分配资源
    for i in range(3):
        try_Available[i] -= resources[i]
        try_Allocation[task][i] += resources[i]
        try_Need[task][i] -= resources[i]

    # 在安全性检查开始前添加一个标记列表
    work = copy.deepcopy(try_Available)
    finished = [False] * x  # 用于标记任务是否已处理
    
    for _ in range(x):
        found = False
        for i in range(x):
            # 检查任务是否已处理且需求是否可以满足
            if not finished[i] and all(try_Need[i][j] <= work[j] for j in range(3)):
                # 找到符合条件的任务
                for j in range(3):
                    work[j] += try_Allocation[i][j]
                finished[i] = True  # 标记该任务已处理
                found = True
                break
        if not found:
            # 没有找到符合条件的任务，返回拒绝
            print(f"请求 {req_num} 被拒绝：分配后系统处于不安全状态。")
            return (x, task, Max, Allocation, Need, Available, False)

    # 循环完毕，表示系统处于安全状态，正式分配资源
    for i in range(3):
        Available[i] -= resources[i]
        Need[task][i] -= resources[i]
        Allocation[task][i] += resources[i]

    print(f"请求 {req_num} 被批准：任务 {task} 的请求 {resources} 已分配。")
    return (x, task, Max, Allocation, Need, Available, True)
def release_resources(Need, Allocation, Available, completed_tasks):
    """
    检查 Need 矩阵中是否有任务的需求全部为零，如果有，则将该任务的资源归还给 Available，
    并记录已经完成的任务。

    参数:
        Need (list of lists): 当前所有任务的剩余需求矩阵。
        Allocation (list of lists): 当前所有任务的资源分配矩阵。
        Available (list): 当前系统可用的资源列表。
        completed_tasks (list, optional): 已完成任务的列表，用于避免重复处理。

    返回:
        tuple:
            Available (list): 更新后的可用资源列表。
            Allocation (list of lists): 更新后的资源分配矩阵。
            newly_completed_tasks (list): 新近完成的任务列表。
    """

    for task_id, need in enumerate(Need):
        # 检查该任务是否已经完成
        if task_id in completed_tasks:
            continue

        # 检查该任务的需求是否全部为零
        if all(n == 0 for n in need):
            # 将该任务分配的资源归还给 Available
            for i in range(len(Available)):
                Available[i] += Allocation[task_id][i]
                Allocation[task_id][i] = 0  # 重置 Allocation

            # 记录该任务为已完成
            newly_completed_tasks.append(task_id)
            print(f"任务 {task_id} 已完成")

    # 更新已完成任务列表
    completed_tasks.extend(newly_completed_tasks)

    return Available, Allocation, newly_completed_tasks
# 读取Max矩阵
def read_max_matrix(filename):
    max_matrix = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # 将每行转换为整数列表
            row = list(map(int, line.strip().split()))
            max_matrix.append(row)
    return max_matrix

# 读取请求序列
def read_requests(filename):
    requests = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # 将每行转换为整数列表
            nums = list(map(int, line.strip().split()))
            # 重构请求格式：[[资源A,资源B,资源C], 任务ID, 请求ID]
            requests.append([nums[:3], nums[3], nums[4]])
    return requests
if __name__ == "__main__":
    # 定义系统资源上限和任务数量
    x = 5  # 任务数量
    OSmax_a = 10  # 资源A的系统最大值
    OSmax_b = 5   # 资源B的系统最大值
    OSmax_c = 7   # 资源C的系统最大值
    newly_completed_tasks = []
    # 读取Max矩阵
    Max = read_max_matrix("max_matrix.txt")
    # 读取请求序列
    requests = read_requests("requests.txt")
    # 初始化 Allocation 矩阵为全0
    Allocation = [[0, 0, 0] for _ in range(x)]

    # 初始化 Need 矩阵为 Max 矩阵的副本
    Need = [task.copy() for task in Max]

    # 初始化 Available 列表为 Osmax
    Available = [OSmax_a, OSmax_b, OSmax_c]

    # Osmax 列表
    OSmax = [OSmax_a, OSmax_b, OSmax_c]
    flag = True
    newly_completed_tasks = []
    print("\n初始状态:")
    print(f"Available: {Available}")
    print(f"Allocation: {Allocation}")
    print(f"Need: {Need}\n")
    for req in requests:
        resources, task, req_num = req
        x, task, Max, Allocation, Need, Available, flag = bankers_algorithm(x, resources, task, Max, Allocation, Need, Available, req_num)
        if flag:
            print(f"当前Available: {Available}")
            print(f"当前Allocation: {Allocation}")
            print(f"当前Need: {Need}\n")
        else:
            print(f"当前Available: {Available}")
            print(f"当前Allocation: {Allocation}")
            print(f"当前Need: {Need}\n")
        Available, Allocation, newly_completed_tasks = release_resources(Need, Allocation, Available, newly_completed_tasks)
        print(f"尝试归还资源后:")
        print(f"可以比较一下归还资源前后的Available")
        print(f"如果Available相同，则没有归还资源")
        print(f"归还资源后的Available: {Available}")
        print(f"Allocation: {Allocation}")
        print(f"Need: {Need}\n")
