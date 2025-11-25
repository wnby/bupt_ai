import pandas as pd
import numpy as np
def load():
    df = pd.read_csv(r'C:\Users\49452\Desktop\神经网络 作业\刘逸-2022212054-A7\AT_8000_M&F.csv')  # 跳过文件的第一行，第二行是列名

    # 初始化结果列表
    result = []

    # 遍历每一行，根据列名读取数据
    for i, row in df.iterrows():
        # 获取“first name” 和 “last name” 列，拼接成 full name
        name1 = row['Tc'] + ' ' + row['Kaplan'] + '<'
        result.append({"number": i, "name": name1})

    # 查看结果
    # print(result)
    return result
def load_JP():
    df = pd.read_csv(r'C:\Users\49452\Desktop\神经网络 作业\刘逸-2022212054-A7\JP_8000_M&F.csv')  # 跳过文件的第一行，第二行是列名

    # 初始化结果列表
    result = []

    # 遍历每一行，根据列名读取数据
    for i, row in df.iterrows():
        # 获取“first name” 和 “last name” 列，拼接成 full name
        name1 = row['Mohammad'] + ' ' + row['Riaz Rahat'] + '<'
        result.append({"number": i, "name": name1})

    # 查看结果
    # print(result)
    return result
def load_CN():
    df = pd.read_csv(r'C:\Users\49452\Desktop\神经网络 作业\刘逸-2022212054-A7\CN_8000_M&F.csv')  # 跳过文件的第一行，第二行是列名

    # 初始化结果列表
    result = []

    # 遍历每一行，根据列名读取数据
    for i, row in df.iterrows():
        # 确保'Ann'列的值不是NaN，并且'Ann'与'Liao'列是字符串类型
        if not pd.isna(row['Ann']) and isinstance(row['Ann'], str) and isinstance(row['Liao'], str):
            name1 = row['Ann'] + ' ' + row['Liao'] + '<'
            result.append({"number": i, "name": name1})

    # 查看结果
    # print(result)
    return result
def load_TR():
    df = pd.read_csv(r'C:\Users\49452\Desktop\神经网络 作业\刘逸-2022212054-A7\TR_8000_M&F.csv')  # 跳过文件的第一行，第二行是列名

    # 初始化结果列表
    result = []

    # 遍历每一行，根据列名读取数据
    for i, row in df.iterrows():
        # 获取“first name” 和 “last name” 列，拼接成 full name
        name1 = row['Hava'] + ' ' + row['Seyis'] + '<'

        result.append({"number": i, "name": name1})

    # 查看结果
    # print(result)
    return result