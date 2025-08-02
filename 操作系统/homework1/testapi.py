import os
import qianfan
import re

os.environ["QIANFAN_AK"] = "pxMuuFHAFuIAX2KIwPcCEYT6"
os.environ["QIANFAN_SK"] = "V1gbd8YXI0OfcgvA2BNf9ao2gq2jPSmi"

def ModeChinese(text,engine=4):
    textahead = r"""我用的是Windows，，请你给出cmd命令行，注意，cmd指令的前后请加两个星号，
    举个例子，**cd**，
    接下来我会描述我的需求：
    """
    engines = ["ernie-4.0-8k-latest","completions_pro","ernie-4.0-8k-preview","ernie-speed-128k","ernie_speed"]
    text = textahead + text
    response = qianfan.ChatCompletion().do(endpoint=engines[3], messages=[{"role": "user", "content": text}])
    result = response['result']
    text = result
    # 使用正则表达式提取所有**...**之间的内容
    code_blocks = re.findall(r"\*\*(.*?)\*\*", result)
    print(result)
    if code_blocks:
        # 对每个匹配到的代码块按双换行符分割，并去除空白
        split_content = [part.strip() for block in code_blocks for part in block.split('\n\n') if part.strip()]
        
        print(split_content)  # 输出分割后的内容列表
        return text,split_content  # 返回分割后的内容列表
    else:
        return text,[]  # 如果没有匹配到内容，返回空列表

if __name__ == '__main__':
    ModeChinese("随便给我几个命令")