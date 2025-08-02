#include <iostream>
#include <chrono>
#include <windows.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <io.h>
#pragma comment(lib, "ws2_32.lib")  // 确保链接 ws2_32.lib
using namespace std;
using namespace std::chrono;

void measure_open_close() {
    
    auto start = high_resolution_clock::now();

    // 打开并关闭一个文件
    HANDLE file = CreateFile(L"test_file.txt", GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file != INVALID_HANDLE_VALUE) {
        CloseHandle(file);
        cout << "1" << endl;
    }

    auto end = high_resolution_clock::now();
    cout << "open/close time: " << duration_cast<microseconds>(end - start).count() << "us" << endl;
}



void measure_file_delete() {


    // 创建文件后删除它
    HANDLE file = CreateFile(L"delete_file.txt", GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    auto start = high_resolution_clock::now();
    if (file != INVALID_HANDLE_VALUE) {
        CloseHandle(file);
        DeleteFile(L"delete_file.txt");
    }

    auto end = high_resolution_clock::now();
    cout << "file delete time: " << duration_cast<microseconds>(end - start).count() << "us" << endl;
}

void write_to_mmap_file()
{
    // 打开文件，创建或覆盖
    HANDLE file = CreateFile(
        L"mmap_file.txt",
        GENERIC_READ | GENERIC_WRITE,
        0,
        NULL,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );

    if (file == INVALID_HANDLE_VALUE) {
        cerr << "Failed to open or create file." << endl;
        return;
    }

    // 创建文件映射对象
    HANDLE mapping = CreateFileMapping(file, NULL, PAGE_READWRITE, 0, 1, NULL); // 映射 1 字节
    if (!mapping) {
        cerr << "Failed to create file mapping." << endl;
        CloseHandle(file);
        return;
    }

    // 映射文件视图到进程地址空间，只映射 1 字节
    void* mapView = MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, 1);
    if (!mapView) {
        cerr << "Failed to map view of file." << endl;
        CloseHandle(mapping);
        CloseHandle(file);
        return;
    }

    // 写入数据 '1'（字符 '1' 的 ASCII 值是 49）
    char* data = static_cast<char*>(mapView);

        for (int j = 0; j < 1024; ++j) {
        for (int i = 0; i < 1024; ++i) {
            data[i] = '1'; // 写入字符 '1' (ASCII 值 49)
        }
    }

    // 解除映射和关闭句柄
    UnmapViewOfFile(mapView);
    CloseHandle(mapping);
    CloseHandle(file);
}
void measure_mmap_latency()
{
    auto start = high_resolution_clock::now(); // 记录开始时间

    // 打开文件
    HANDLE file = CreateFile(L"mmap_file.txt", GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) {
        cerr << "Failed to open file." << endl;
        return;
    }

    // 获取文件大小
    LARGE_INTEGER fileSize;
    if (!GetFileSizeEx(file, &fileSize)) {
        cerr << "Failed to get file size." << endl;
        CloseHandle(file);
        return;
    }

    // 创建文件映射对象
    HANDLE mapping = CreateFileMapping(file, NULL, PAGE_READWRITE, 0, 0, NULL);
    if (!mapping) {
        cerr << "Failed to create file mapping." << endl;
        CloseHandle(file);
        return;
    }

    // 映射文件视图到进程地址空间
    void* mapView = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, fileSize.QuadPart);
    if (!mapView) {
        cerr << "Failed to map view of file." << endl;
        CloseHandle(mapping);
        CloseHandle(file);
        return;
    }

    // 读取映射区域数据
    char* data = static_cast<char*>(mapView);
    long long sum = 0;
    for (LONGLONG i = 0; i < fileSize.QuadPart; i++) {
        sum += data[i]; // 简单累加，模拟读取操作
    }

    // 解除映射和关闭句柄
    UnmapViewOfFile(mapView);
    CloseHandle(mapping);
    CloseHandle(file);

    auto end = high_resolution_clock::now(); // 记录结束时间
    std::cout << "Sum: " << sum << ", mmap read and sum time: " << duration_cast<microseconds>(end - start).count() << "us" << endl; // 输出结果
}

int main() {
    // 初始化 Windows 网络


    measure_open_close();
    measure_file_delete();

    write_to_mmap_file();
    measure_mmap_latency();


    return 0;
}