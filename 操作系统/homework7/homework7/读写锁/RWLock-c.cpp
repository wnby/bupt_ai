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
        std::cout << "[���� " << readerId << "] �ȴ���ȡ\n";
        while (currentWriters > 0 || pendingWriters > 0) {
            std::cout << "[���� " << readerId << "] �ȴ�д���ͷ�\n";
            conditionVar.wait(lock);
        }
        pendingReaders--;
        currentReaders++;
        std::cout << "[���� " << readerId << "] ��ʼ��ȡ\n";
    }

    void finishRead(int readerId) {
        std::unique_lock<std::mutex> lock(mutexLock);
        currentReaders--;
        std::cout << "[���� " << readerId << "] ��ȡ���\n";
        if (currentReaders == 0 && pendingWriters > 0) {
            std::cout << "[���� " << readerId << "] ֪ͨд��\n";
            conditionVar.notify_all();
        }
    }

    void requestWrite(int writerId) {
        std::unique_lock<std::mutex> lock(mutexLock);
        pendingWriters++;
        std::cout << "[д�� " << writerId << "] �ȴ�д��\n";
        while (currentReaders > 0 || currentWriters > 0) {
            std::cout << "[д�� " << writerId << "] �ȴ����ߺ�д���ͷ�\n";
            conditionVar.wait(lock);
        }
        pendingWriters--;
        currentWriters++;
        std::cout << "[д�� " << writerId << "] ��ʼд��\n";
    }

    void finishWrite(int writerId) {
        std::unique_lock<std::mutex> lock(mutexLock);
        currentWriters--;
        std::cout << "[д�� " << writerId << "] д�����\n";
        if (pendingWriters > 0) {
            std::cout << "[д�� " << writerId << "] ֪ͨд��\n";
            conditionVar.notify_all();
        }
        else if (pendingReaders > 0) {
            std::cout << "[д�� " << writerId << "] ֪ͨ����\n";
            conditionVar.notify_all();
        }
    }
};

std::atomic<int> dataStore(0);  // ������Դ
std::atomic<bool> terminateFlag(false);  // �����߳�ֹͣ�ı�־

// ���̺߳���
void readerThread(RWLock& lockManager, int readerId) {
    while (!terminateFlag.load()) {
        lockManager.requestRead(readerId);
        std::cout << "[���� " << readerId << "] ���ڶ�ȡ: " << dataStore.load() << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        lockManager.finishRead(readerId);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "[���� " << readerId << "] ��ֹͣ\n";
}

// д�̺߳���
void writerThread(RWLock& lockManager, int writerId) {
    while (!terminateFlag.load()) {
        lockManager.requestWrite(writerId);
        dataStore++;
        std::cout << "[д�� " << writerId << "] ����д��: " << dataStore.load() << "\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        lockManager.finishWrite(writerId);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }
    std::cout << "[д�� " << writerId << "] ��ֹͣ\n";
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

    std::cout << "�����߳��ѽ���\n";
    return 0;
}
