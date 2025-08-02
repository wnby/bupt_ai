import random
import time
import copy
import os
def generate_bankers_random_requests(x, OSmax_a, OSmax_b, OSmax_c):
    """
    生成用于银行家算法的请求序列。

    参数:
        x (int): 任务数量。
        OSmax_a (int): 资源A的系统最大值。
        OSmax_b (int): 资源B的系统最大值。
        OSmax_c (int): 资源C的系统最大值。

    返回:
        list: 请求列表，每个元素包括所需资源数、任务编号和请求编号。
    """
    # 步骤 1: 创建 Max 矩阵（每个任务的最大资源需求）
    Max = []
    i = 0
    while i != x:
        max_a = random.randint(0, OSmax_a)
        max_b = random.randint(0, OSmax_b)
        max_c = random.randint(0, OSmax_c)
        if max_a == 0 and max_b == 0 and max_c == 0:
            continue
        i = i + 1
        Max.append([max_a, max_b, max_c])
    print("Max 矩阵（每个任务的最大资源需求）:")
    for i, m in enumerate(Max):
        print(f"任务 {i}: {m}")

    # 步骤 2: 初始化标志位列表（标记任务是否完成所有请求）
    flag = [0] * x

    # 步骤 3: 初始化 Need 矩阵为 Max 矩阵的副本
    Need = [task.copy() for task in Max]

    # 步骤 4: 生成请求列表
    request_list = []
    loop_count = 0  # 循环计数器，用于标记请求编号

    while not all(f == 1 for f in flag):
        # 随机选择一个任务
        n = random.randint(0, x - 1)

        # 获取该任务的当前需求
        need_a, need_b, need_c = Need[n]

        # 如果该任务的需求都为0，标记为已完成
        if need_a == 0 and need_b == 0 and need_c == 0:
            flag[n] = 1
            continue

        # 生成一个非零的请求
        while True:
            request_a = random.randint(0, need_a)
            request_b = random.randint(0, need_b)
            request_c = random.randint(0, need_c)
            if request_a != 0 or request_b != 0 or request_c != 0:
                break
        loop_count += 1
        # 将请求添加到请求列表中
        request_list.append([[request_a, request_b, request_c], n, loop_count])
        print(f"请求 {loop_count}: 任务 {n} 请求资源 {request_a, request_b, request_c}")

        # 更新 Need 矩阵
        Need[n][0] -= request_a
        Need[n][1] -= request_b
        Need[n][2] -= request_c

        
    return Max, request_list
def generate_bankers_safe_requests(x, OSmax_a, OSmax_b, OSmax_c): 
    """
    生成用于银行家算法的请求序列。

    参数:
        x (int): 任务数量。
        OSmax_a (int): 资源A的系统最大值。
        OSmax_b (int): 资源B的系统最大值。
        OSmax_c (int): 资源C的系统最大值。

    返回:
        Max (list): 每个任务的最大资源需求矩阵。
        request_list (list): 请求列表，每个元素包括所需资源数、任务编号和请求编号。
    """
    # 初始化系统可用资源（Available）
    Available = [OSmax_a, OSmax_b, OSmax_c]
    
    # 初始化 Max 矩阵为每个任务的 [0, 0, 0]
    Max = [[0, 0, 0] for _ in range(x)]
    
    # 初始化标志位列表（标记任务是否完成）
    flag = [0] * x

    # 初始化请求列表
    request_list = []
    loop_count = 0  # 请求计数器，用于标记请求编号

    while not all(f == 1 for f in flag):
        # 如果系统资源耗尽，退出循环
        if all(avail == 0 for avail in Available):
            print("系统资源耗尽，无法继续生成请求。")
            break

        # 随机选择一个未完成的任务
        unfinished_tasks = [i for i, f in enumerate(flag) if f == 0]
        if not unfinished_tasks:
            break
        n = random.choice(unfinished_tasks)

        # 生成一个非零的请求，���源量不超过系统当前可用资源
        while True:
            request_a = random.randint(0, Available[0])
            request_b = random.randint(0, Available[1])
            request_c = random.randint(0, Available[2])
            if request_a != 0 or request_b != 0 or request_c != 0:
                break

        # 更新系统可用资源（Available）
        Available[0] -= request_a
        Available[1] -= request_b
        Available[2] -= request_c

        # 更新任务的最大资源需求（Max）
        Max[n][0] += request_a
        Max[n][1] += request_b
        Max[n][2] += request_c

        loop_count += 1
        # 将请求添加到请求列表中
        request_list.append([[request_a, request_b, request_c], n, loop_count])
        print(f"请求 {loop_count}: 任务 {n} 请求资源 {request_a, request_b, request_c}")

        # 如果系统可用资源为零，标记任务为已完成
        if all(avail == 0 for avail in Available):
            flag[n] = 1

    # 输出每个任务的最大资源需求（Max 矩阵）
    print("\nMax 矩阵（每个任务的最大资源需求）:")
    for i, m in enumerate(Max):
        print(f"任务 {i}: {m}")

    return Max, request_list
# ... 前面的代码保持不变 ...

if __name__ == "__main__":
    # 定义系统资源上限和任务数量
    x = 5  # 任务数量
    OSmax_a = 10  # 资源A的系统最大值
    OSmax_b = 5   # 资源B的系统最大值
    OSmax_c = 7   # 资源C的系统最大值
    newly_completed_tasks = []
    
    # 使用相对路径
    max_file = "max_matrix.txt"
    requests_file = "requests.txt"
    
    # 生成请求序列和 Max 矩阵
    Max, requests = generate_bankers_random_requests(x, OSmax_a, OSmax_b, OSmax_c)
    
    try:
        # 将Max矩阵写入文件
        with open(max_file, 'w', encoding='utf-8') as f:
            for row in Max:
                f.write(f"{row[0]} {row[1]} {row[2]}\n")
        print(f"Max矩阵已保存到: {max_file}")
        
        # 将请求序列写入文件
        with open(requests_file, 'w', encoding='utf-8') as f:
            for request in requests:
                resources, task_id, request_id = request
                f.write(f"{resources[0]} {resources[1]} {resources[2]} {task_id} {request_id}\n")
        print(f"请求序列已保存到: {requests_file}")
        
    except Exception as e:
        print(f"保存文件时出错: {e}")



