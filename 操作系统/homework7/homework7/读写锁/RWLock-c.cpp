#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cassert>

class RWLock {
private:
    std::mutex mutexLock;
    std::condition_variable conditionVar;
    int currentReaders;
    int currentWriters;
    int pendingReaders;
    int pendingWriters;

public:
    RWLock() : currentReaders(0), currentWriters(0), pendingReaders(0), pendingWriters(0) {}

    void requestRead(int readerId) {
        std::unique_lock<std::mutex> lock(mutexLock);
        pendingReaders++;
        std::cout << "[读者 " << readerId << "] 等待读取\n";
        while (currentWriters > 0 || pendingWriters > 0) {
            std::cout << "[读者 " << readerId << "] 等待写者释放\n";
            conditionVar.wait(lock);
        }
        pendingReaders--;
        currentReaders++;
        std::cout << "[读者 " << readerId << "] 开始读取\n";
    }

    void finishRead(int readerId) {
        std::unique_lock<std::mutex> lock(mutexLock);
        currentReaders--;
        std::cout << "[读者 " << readerId << "] 读取完成\n";
        if (currentReaders == 0 && pendingWriters > 0) {
            std::cout << "[读者 " << readerId << "] 通知写者\n";
            conditionVar.notify_all();
        }
    }

    void requestWrite(int writerId) {
        std::unique_lock<std::mutex> lock(mutexLock);
        pendingWriters++;
        std::cout << "[写者 " << writerId << "] 等待写入\n";
        while (currentReaders > 0 || currentWriters > 0) {
            std::cout << "[写者 " << writerId << "] 等待读者和写者释放\n";
            conditionVar.wait(lock);
        }
        pendingWriters--;
        currentWriters++;
        std::cout << "[写者 " << writerId << "] 开始写入\n";
    }

    void finishWrite(int writerId) {
        std::unique_lock<std::mutex> lock(mutexLock);
        currentWriters--;
        std::cout << "[写者 " << writerId << "] 写入完成\n";
        if (pendingWriters > 0) {
            std::cout << "[写者 " << writerId << "] 通知写者\n";
            conditionVar.notify_all();
        }
        else if (pendingReaders > 0) {
            std::cout << "[写者 " << writerId << "] 通知读者\n";
            conditionVar.notify_all();
        }
    }
};

std::atomic<int> dataStore(0);  // 共享资源
std::atomic<bool> terminateFlag(false);  // 控制线程停止的标志

// 读线程函数
void readerThread(RWLock& lockManager, int readerId) {
    while (!terminateFlag.load()) {
        lockManager.requestRead(readerId);
        std::cout << "[读者 " << readerId << "] 正在读取: " << dataStore.load() << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        lockManager.finishRead(readerId);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "[读者 " << readerId << "] 已停止\n";
}

// 写线程函数
void writerThread(RWLock& lockManager, int writerId) {
    while (!terminateFlag.load()) {
        lockManager.requestWrite(writerId);
        dataStore++;
        std::cout << "[写者 " << writerId << "] 正在写入: " << dataStore.load() << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        lockManager.finishWrite(writerId);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    std::cout << "[写者 " << writerId << "] 已停止\n";
}

int main() {
    RWLock lockManager;
    std::vector<std::thread> threadList;
    int readerCount = 2;
    int writerCount = 2;

    for (int i = 0; i < readerCount; ++i) {
        threadList.emplace_back(readerThread, std::ref(lockManager), i + 1);
    }
    for (int i = 0; i < writerCount; ++i) {
        threadList.emplace_back(writerThread, std::ref(lockManager), i + 1);
    }

    std::this_thread::sleep_for(std::chrono::seconds(5));
    terminateFlag.store(true);

    for (auto& t : threadList) {
        if (t.joinable()) {
            t.join();
        }
    }

    std::cout << "所有线程已结束\n";
    return 0;
}
