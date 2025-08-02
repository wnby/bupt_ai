import time
import random
import threading

class RWLock:
    def __init__(self):
        # 初始化状态
        self.readers = 0  # 当前正在读取的读者数量
        self.writers = 0  # 当前正在写的写者数量
        self.pending_readers = 0  # 等待的读者数量
        self.pending_writers = 0  # 等待的写者数量
        self.lock = threading.Lock()

    def _compare_and_swap(self, address, expected, new_value):
        """模拟CAS操作，比较并交换。"""
        with self.lock:
            if address == expected:
                return new_value
            return address
    
    def request_read(self, reader_id):
        """请求读取资源。"""
        print(f"[读者 {reader_id}] 请求读取")
        self.pending_readers += 1
        while True:
            if self.writers == 0 and self.pending_writers == 0:
                # 如果没有写者和等待的写者，允许读
                self.readers += 1
                self.pending_readers -= 1
                print(f"[读者 {reader_id}] 开始读取")
                break
            else:
                # 如果有写者或等待写者，继续等待
                print(f"[读者 {reader_id}] 等待写者释放")
                time.sleep(0.1)  # 自旋等待

    def finish_read(self, reader_id):
        """完成读取资源。"""
        self.readers -= 1
        print(f"[读者 {reader_id}] 完成读取")
        if self.readers == 0 and self.pending_writers > 0:
            print(f"[读者 {reader_id}] 通知写者")
            time.sleep(0.1)

    def request_write(self, writer_id):
        """请求写资源。"""
        print(f"[写者 {writer_id}] 请求写入")
        self.pending_writers += 1
        while True:
            if self.readers == 0 and self.writers == 0:
                # 如果没有读者和写者，允许写
                self.writers += 1
                self.pending_writers -= 1
                print(f"[写者 {writer_id}] 开始写入")
                break
            else:
                # 如果有读者或写者，继续等待
                print(f"[写者 {writer_id}] 等待读者和写者释放")
                time.sleep(0.1)  # 自旋等待

    def finish_write(self, writer_id):
        """完成写操作。"""
        self.writers -= 1
        print(f"[写者 {writer_id}] 完成写入")
        if self.pending_writers > 0:
            print(f"[写者 {writer_id}] 通知写者")
            time.sleep(0.1)
        elif self.pending_readers > 0:
            print(f"[写者 {writer_id}] 通知读者")
            time.sleep(0.1)

dataStore = 0
terminateFlag = False

def reader_thread(lockManager, reader_id):
    global dataStore
    while not terminateFlag:
        lockManager.request_read(reader_id)
        print(f"[读者 {reader_id}] 正在读取: {dataStore}")
        time.sleep(random.uniform(0.1, 0.3))
        lockManager.finish_read(reader_id)
        time.sleep(random.uniform(0.1, 0.3))
    print(f"[读者 {reader_id}] 已停止")

def writer_thread(lockManager, writer_id):
    global dataStore
    while not terminateFlag:
        lockManager.request_write(writer_id)
        dataStore += 1
        print(f"[写者 {writer_id}] 正在写入: {dataStore}")
        time.sleep(random.uniform(0.1, 0.3))
        lockManager.finish_write(writer_id)
        time.sleep(random.uniform(0.1, 0.3))
    print(f"[写者 {writer_id}] 已停止")

if __name__ == "__main__":
    lockManager = RWLock()
    threadList = []
    readerCount = 2
    writerCount = 2

    # 创建并启动读者线程
    for i in range(readerCount):
        threadList.append(threading.Thread(target=reader_thread, args=(lockManager, i + 1)))
    
    # 创建并启动写者线程
    for i in range(writerCount):
        threadList.append(threading.Thread(target=writer_thread, args=(lockManager, i + 1)))

    # 启动所有线程
    for thread in threadList:
        thread.start()

    # 模拟运行5秒钟，然后停止
    time.sleep(5)
    terminateFlag = True

    # 等待所有线程结束
    for thread in threadList:
        thread.join()

    print("所有线程已结束")
