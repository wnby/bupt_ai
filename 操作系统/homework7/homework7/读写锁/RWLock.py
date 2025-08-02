import threading
import time
import random

class RWLock:
    def __init__(self):
        self.mutexLock = threading.Lock()
        self.conditionVar = threading.Condition(self.mutexLock)
        self.currentReaders = 0
        self.currentWriters = 0
        self.pendingReaders = 0
        self.pendingWriters = 0

    def requestRead(self, readerId):
        with self.mutexLock:
            self.pendingReaders += 1
            print(f"[读者 {readerId}] 等待读取")
            while self.currentWriters > 0 or self.pendingWriters > 0:#写者或等待写者
                print(f"[读者 {readerId}] 等待写者释放")
                self.conditionVar.wait()
            self.pendingReaders -= 1
            self.currentReaders += 1
            print(f"[读者 {readerId}] 开始读取")

    def finishRead(self, readerId):
        with self.mutexLock:
            self.currentReaders -= 1
            print(f"[读者 {readerId}] 读取完成")
            if self.currentReaders == 0 and self.pendingWriters > 0:
                print(f"[读者 {readerId}] 通知写者")
                self.conditionVar.notify_all()

    def requestWrite(self, writerId):
        with self.mutexLock:
            self.pendingWriters += 1
            print(f"[写者 {writerId}] 等待写入")
            while self.currentReaders > 0 or self.currentWriters > 0:#
                print(f"[写者 {writerId}] 等待读者和写者释放")
                self.conditionVar.wait()
            self.pendingWriters -= 1
            self.currentWriters += 1
            print(f"[写者 {writerId}] 开始写入")

    def finishWrite(self, writerId):
        with self.mutexLock:
            self.currentWriters -= 1
            print(f"[写者 {writerId}] 写入完成")
            if self.pendingWriters > 0:
                print(f"[写者 {writerId}] 通知写者")
                self.conditionVar.notify_all()
            elif self.pendingReaders > 0:
                print(f"[写者 {writerId}] 通知读者")
                self.conditionVar.notify_all()

dataStore = 0
terminateFlag = False

def readerThread(lockManager, readerId):
    global dataStore
    while not terminateFlag:
        lockManager.requestRead(readerId)
        print(f"[读者 {readerId}] 正在读取: {dataStore}")
        time.sleep(random.uniform(0.1, 0.3))
        lockManager.finishRead(readerId)
        time.sleep(random.uniform(0.1, 0.3))
    print(f"[读者 {readerId}] 已停止")

def writerThread(lockManager, writerId):
    global dataStore
    while not terminateFlag:
        lockManager.requestWrite(writerId)
        dataStore += 1
        print(f"[写者 {writerId}] 正在写入: {dataStore}")
        time.sleep(random.uniform(0.1, 0.3))
        lockManager.finishWrite(writerId)
        time.sleep(random.uniform(0.1, 0.3))
    print(f"[写者 {writerId}] 已停止")

if __name__ == "__main__":
    lockManager = RWLock()
    threadList = []
    readerCount = 2
    writerCount = 2

    for i in range(readerCount):
        threadList.append(threading.Thread(target=readerThread, args=(lockManager, i + 1)))
    
    for i in range(writerCount):
        threadList.append(threading.Thread(target=writerThread, args=(lockManager, i + 1)))

    for thread in threadList:
        thread.start()

    time.sleep(5)
    terminateFlag = True

    for thread in threadList:
        thread.join()

    print("所有线程已结束")
