import os
import time
import socket
import mmap
import select

# 测量时间函数
def measure_time(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return end - start, result

# 1. 测量 open clos 的时间
def open_close_time(file_path):
    f = open(file_path, 'w')
    f.close()
    def open_close():
        f = open(file_path, 'r+b')
        f.close()
    return measure_time(open_close)

# 3. 测量 0K File Delete 的时间
def file_delete_time(file_path):
    # 创建一个 0K 文件
    with open(file_path, 'w'):
        pass
    def create_and_delete_file():
        # 删除文件
        os.unlink(file_path)
    return measure_time(create_and_delete_file)

# 4. 测量 Mmap Latency
def mmap_latency_time(file_path, size):
    # Step 1: 创建并写入一个指定大小的文件
    with open(file_path, 'wb') as f:
        f.write(b'\x01' * size)  # 写入全是1的字节作为示例内容

    # Step 2: 将文件 mmap 到内存中，并从内存中读取文件
    def mmap_read():
        with open(file_path, 'r+b') as f:
            # 使用 mmap 映射文件
            mm = mmap.mmap(f.fileno(), size)
            # 从内存中读取内容并计算字节和
            total_sum = sum(mm[i] for i in range(size))
            mm.close()  # 关闭内存映射
        return 0
    return measure_time(mmap_read)

# 主程序
if __name__ == "__main__":
    file_path = "test_file.txt"
    mmap_size = 1024*1024
    iterations = 100   # 执行次数

    # 用于累加时间的变量
    open_close_total = 0
    file_delete_total = 0
    mmap_latency_total = 0

    for _ in range(iterations):
        # 1. open close 时间
        open_close_duration, _ = open_close_time(file_path)
        open_close_total += open_close_duration

        # 2. 0K File Delete 时间
        file_delete_duration, _ = file_delete_time(file_path)
        file_delete_total += file_delete_duration

        # 3. Mmap Latency 时间
        mmap_latency_duration, _ = mmap_latency_time(file_path, mmap_size)
        mmap_latency_total += mmap_latency_duration

    # 计算平均时间
    open_close_average = open_close_total / iterations
    file_delete_average = file_delete_total / iterations
    mmap_latency_average = mmap_latency_total / iterations

    # 打印平均时间
    print(f"Average open close time: {open_close_average:.10f} seconds")
    print(f"Average 0K File Delete time: {file_delete_average:.10f} seconds")
    print(f"Average Mmap Latency time: {mmap_latency_average:.10f} seconds")