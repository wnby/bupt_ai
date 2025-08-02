#include <iostream>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

using namespace std;
using namespace std::chrono;

void measure_open_close() {
    auto start = high_resolution_clock::now();

    // 打开并关闭一个文件
    int file = open("test_file.txt", O_RDWR);
    if (file != -1) {
        close(file);
        cout << "1" << endl;
    }

    auto end = high_resolution_clock::now();
    cout << "open/close time: " << duration_cast<microseconds>(end - start).count() << "us" << endl;
}

void measure_file_delete() {
    // 创建文件后删除它
    int file = open("delete_file.txt", O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    auto start = high_resolution_clock::now();
    if (file != -1) {
        close(file);
        unlink("delete_file.txt"); // 删除文件
    }

    auto end = high_resolution_clock::now();
    cout << "file delete time: " << duration_cast<microseconds>(end - start).count() << "us" << endl;
}

void write_to_mmap_file() {
    // 打开文件，创建或覆盖
    int file = open("mmap_file.txt", O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (file == -1) {
        cerr << "Failed to open or create file." << endl;
        return;
    }

    // 扩展文件到 1MB 大小
    if (ftruncate(file, 1024 * 1024) == -1) {
        cerr << "Failed to set file size." << endl;
        close(file);
        return;
    }

    // 映射文件视图到进程地址空间
    void* mapView = mmap(NULL, 1024 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0);
    if (mapView == MAP_FAILED) {
        cerr << "Failed to map view of file." << endl;
        close(file);
        return;
    }

    // 写入数据 '1'（字符 '1' 的 ASCII 值是 49）
    char* data = static_cast<char*>(mapView);
    for (int j = 0; j < 1024; ++j) {
        for (int i = 0; i < 1024; ++i) {
            data[i] = '1'; // 写入字符 '1' (ASCII 值 49)
        }
    }

    // 解除映射和关闭文件
    munmap(mapView, 1024 * 1024);
    close(file);
}

void measure_mmap_latency() {
    

    // 打开文件
    int file = open("mmap_file.txt", O_RDWR);
    if (file == -1) {
        cerr << "Failed to open file." << endl;
        return;
    }

    // 获取文件大小
    struct stat sb;
    if (fstat(file, &sb) == -1) {
        cerr << "Failed to get file size." << endl;
        close(file);
        return;
    }
    auto start = high_resolution_clock::now(); // 记录开始时间
    // 映射文件视图到进程地址空间
    void* mapView = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, file, 0);
    if (mapView == MAP_FAILED) {
        cerr << "Failed to map view of file." << endl;
        close(file);
        return;
    }

    // 读取映射区域数据
    char* data = static_cast<char*>(mapView);
    long long sum = 0;
    for (off_t i = 0; i < sb.st_size; i++) {
        sum += data[i]; // 简单累加，模拟读取操作
    }

    // 解除映射和关闭文件
    munmap(mapView, sb.st_size);
    close(file);

    auto end = high_resolution_clock::now(); // 记录结束时间
    std::cout << "Sum: " << sum << ", mmap read and sum time: " << duration_cast<microseconds>(end - start).count() << "us" << endl; // 输出结果
}

int main() {
    measure_open_close();
    measure_file_delete();

    write_to_mmap_file();
    measure_mmap_latency();

    return 0;
}