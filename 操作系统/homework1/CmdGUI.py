import subprocess
import threading
import queue
import customtkinter as ctk
import re
from PIL import Image
from testapi import ModeChinese
import os
class Cmd:
    def __init__(self, mode="English"):
        self.master = ctk.CTk()  # 创建窗口实例
        self.process = None
        self.mode = mode
        self.model = 4
        self.output_queue = queue.Queue()
        self.output_mode = None
        self.commands = []
        self.history = False
        # 设置窗口标题
        self.master.title("CMD GUI")

        # 设置界面
        self.setup_gui()

        # 启动 shell
        self.start_shell()

        # 开始定期检查输出队列
        self.master.after(100, self.update_output)

        # 绑定窗口关闭事件
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        """设置 GUI 布局"""
        self.cmd_frame = ctk.CTkFrame(self.master)
        self.cmd_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # 配置 cmd_frame 的列和行使其可以扩展
        self.cmd_frame.grid_rowconfigure(0, weight=1, uniform="equal")  # Output Text row can expand
        self.cmd_frame.grid_rowconfigure(1, weight=0)  # Input Text row doesn't need to expand
        self.cmd_frame.grid_columnconfigure(0, weight=1)  # Output Text column can expand
        self.cmd_frame.grid_columnconfigure(1, weight=0)  # Input frame column doesn't need to expand
        self.cmd_frame.grid_columnconfigure(2, weight=0)  # Send button column doesn't need to expand

        # 输出框
        self.output_text = ctk.CTkTextbox(self.cmd_frame, width=750, height=350)
        self.output_text.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.output_text.configure(state="disabled")  # 禁止手动编辑

        # 用户输入框
        self.input_frame = ctk.CTkFrame(self.cmd_frame)
        self.input_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")  # Stretch horizontally
        self.input_frame.grid_columnconfigure(0, weight=0)  # Image label column doesn't expand
        self.input_frame.grid_columnconfigure(1, weight=1)  # Input text expands horizontally
        self.input_frame.grid_columnconfigure(2, weight=0)  # Button column doesn't expand

        # 创建图片
        image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_images")
        self.img = ctk.CTkImage(light_image=Image.open(os.path.join(image_path, "chat_dark.png")),
                                dark_image=Image.open(os.path.join(image_path, "chat_light.png")), size=(20, 20))

        # 图片显示在输入框左边
        self.image_label = ctk.CTkLabel(self.input_frame, image=self.img, text="")
        self.image_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # 输入框
        self.input_text = ctk.CTkEntry(self.input_frame, width=600, placeholder_text="输入 CMD 命令...")
        self.input_text.grid(row=0, column=1, padx=10, pady=10, sticky="ew")  # Expand horizontally
        self.input_text.bind("<Return>", self.shot_cmd_button)

        # 发送按钮
        self.send_button = ctk.CTkButton(self.input_frame, text="运行", command=self.shot_cmd_button)
        self.send_button.grid(row=0, column=2, padx=10, pady=10, sticky="e")

        # 当前模式显示 Label
        self.model_label = ctk.CTkLabel(self.cmd_frame, text=f"当前模型: ERNIE-Speed-8K", anchor="w")
        self.model_label.grid(row=2, column=0, padx=20, pady=10, sticky="w")

        # 当前模式显示 Label
        self.mode_label = ctk.CTkLabel(self.cmd_frame, text=f"当前模式: {self.mode}", anchor="w")
        self.mode_label.grid(row=3, column=0, padx=20, pady=10, sticky="w")

    def start_shell(self):
        """启动一个持久的 shell 会话"""
        if self.process is None or self.process.poll() is not None:
            self.process = subprocess.Popen(
                "cmd.exe",
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # 行缓冲
                universal_newlines=True
            )

            # 启动线程读取 stdout 和 stderr
            threading.Thread(target=self.read_stdout, daemon=True).start()
            threading.Thread(target=self.read_stderr, daemon=True).start()
        else:
            print("Shell 已经在运行")

    def read_stdout(self):
        """读取标准输出并放入队列"""
        try:
            while True:  # 一直读取，直到进程结束
                
                line = self.process.stdout.readline()
                if line == '' and self.process.poll() is not None:  # 如果进程结束
                    break
                if line:
                    self.output_queue.put(line)
                    print("stdout:", line.strip())
        except Exception as e:
            self.output_queue.put(f"错误: {str(e)}\n")

    def read_stderr(self):
        """读取标准错误并放入队列"""
        try:
            while True:
                line = self.process.stderr.readline()
                if line == '' and self.process.poll() is not None:  # 如果进程结束
                    break
                if line:
                    if self.output_mode == True:
                        self.output_queue.put(f"错误: {line}")
                        print("stderr:", line.strip())
                    else:
                        self.cd = line
        except Exception as e:
            self.output_queue.put(f"错误: {str(e)}\n")

    def shot_cmd_button(self, event=None):
        """点击按钮或回车时执行命令"""
        cmd = self.input_text.get()

        if cmd.strip() == "":
            return
        self.input_text.delete(0, "end")
        if self.history == True:
            if cmd != "cmdset historyshow" and cmd != "cmdset nohistory" and cmd != "cmdset history":
                with open("command_history.txt", "a") as history_file:
                    history_file.write(cmd + "\n")  # Write the command followed by a newline
        note = re.match(r'^\S+', cmd)
        note = note.group()
        # print(note)
        if note != "cmdset":
            # print(note)
            if self.mode == "English":
                self.send_command(cmd)
            if self.mode == "Chinese":
                text,Commands = ModeChinese(cmd)
                self.display_Commands(Commands,text)
                

        
        else:
            self.display_output(f"输入命令>{cmd}\n")
            note = re.match(r'^\S+\s+(.*)', cmd)
            if note:
                note = note.group(1)
                if note == "help":
                    self.display_help()
                if note == "mode":
                    self.change_ai_mode()
                    self.display_output(f"已切换模式为{self.mode}\n")
                    if self.mode == "Chinese":
                        self.display_output(f"输入的话自动连接AI，连接网络可能需要几秒\n")
                        self.display_output(f"点击Apply以运用指令，点击Reject拒绝所有指令，指令可以编辑\n")
                    if self.mode == "English":
                        self.display_output(f"退出AI联动模式\n")
                if note == "light":
                    current_mode = ctk.get_appearance_mode()
                    if current_mode == "Light":
                        ctk.set_appearance_mode("Dark")
                    else:
                        ctk.set_appearance_mode("Light")
                    self.display_output(f"更改界面主题成功\n")
                match = re.match(r"([a-zA-Z]+)(\d+)", note)
                if match:
                    english_part = match.group(1)
                    number_part = match.group(2)
                    
                    if english_part == "model":
                        if number_part == '1':
                            print(number_part)
                            self.model = 1
                        if number_part == '2':
                            self.model = 2
                        if number_part == '3':
                            self.model = 3
                        if number_part == '4':
                            self.model = 4
                        if number_part == '5':
                            self.model = 5
                        model = self.update_model_label()
                        self.display_output(f"当前模型：{model}\n")
                if note == "history":
                    self.display_output(f"已开启命令保存模式,会记录你在GUI输入过的命令\n")
                    self.history = True
                if note == "nohistory":
                    self.display_output(f"已关闭命令保存模式\n")
                    self.history = False
                if note == "historyshow":
                    try:
                        with open("command_history.txt", "r") as history_file:
                            history_content = history_file.read()
                        if history_content:
                            self.display_output(f"历史记录:\n{history_content}\n")
                        else:
                            self.display_output("历史记录为空。\n")
                    except FileNotFoundError:
                        self.display_output("历史记录文件不存在。\n")

            else:
                self.display_output("在cmdsettings后输入一点东西吧\n")
    def send_command(self, cmd):
        """向 shell 发送命令"""
        try:
            # 发送命令
            self.process.stdin.write(f"{cmd}\n")
            self.process.stdin.flush()
        except Exception as e:
            self.output_queue.put(f"错误: {str(e)}\n")

    def display_output(self, text):
        """在输出框中显示文本"""
        self.output_text.configure(state="normal")
        self.output_text.insert("end", text)
        self.output_text.configure(state="disabled")
        self.output_text.yview("end")

    def update_output(self):
        """从队列中获取输出并显示在 GUI 中"""
        while not self.output_queue.empty():
            try:
                line = self.output_queue.get_nowait()
                self.display_output(line)
            except queue.Empty:
                pass
        self.master.after(100, self.update_output)

    def update_mode_label(self, mode):
        """更新当前目录显示"""
        self.mode_label.configure(text=f"当前模式: {mode}")
    def update_model_label(self):
        """更新当前目录显示"""
        list=["ERNIE-4.0-8K-Latest","ERNIE-4.0-8K","ERNIE-4.0-8K-Preview","ERNIE-Speed-128K","ERNIE-Speed-8K"]
        self.model_label.configure(text=f"当前模型: {list[self.model-1]}")
        return list[self.model-1]
    def on_closing(self):
        """处理窗口关闭事件，确保子进程被终止"""
        if self.process:
            try:
                self.process.stdin.write("exit\n")
                self.process.stdin.flush()
                self.process.terminate()
            except Exception:
                pass
        self.master.destroy()

    def run(self):
        """启动应用"""
        self.master.mainloop()
    def display_Commands(self, Commands, text):
        """展示 Commands 列表并提供编辑功能，同时展示新窗口显示 text 内容"""
        # 创建新的窗口
        edit_window = ctk.CTkToplevel(self.master)
        edit_window.title("编辑 Commands")
        
        # 存储所有文本框对象
        textboxes = []

        # 配置列和行的扩展性
        edit_window.grid_columnconfigure(0, weight=1, uniform="equal")
        edit_window.grid_columnconfigure(1, weight=2, uniform="equal")  # 第二列宽度更大，容纳文本框
        edit_window.grid_rowconfigure(len(Commands), weight=1)

        # 创建文本框显示 Commands 列表
        for idx, command in enumerate(Commands):
            label = ctk.CTkLabel(edit_window, text=f"命令 {idx + 1}:")
            label.grid(row=idx, column=0, padx=10, pady=5, sticky="w")

            textbox = ctk.CTkTextbox(edit_window, height=3, width=70)
            textbox.grid(row=idx, column=1, padx=10, pady=5, sticky="ew")  # 让文本框扩展填充可用空间
            textbox.insert("end", command)  # 显示原始命令

            textboxes.append(textbox)

        # Apply 按钮
        def apply_changes():
            self.commands = [textbox.get("1.0", "end-1c") for textbox in textboxes]
            for command in self.commands:
                if command:
                    self.send_command(command)
            edit_window.destroy()
            self.commands = []
            self.display_output("\n".join(self.commands))

        # Reject 按钮
        def reject_changes():
            edit_window.destroy()
            self.display_output("操作已取消，返回空命令列表。\n")
            print([])  # 返回空列表

        apply_button = ctk.CTkButton(edit_window, text="Apply", command=apply_changes)
        apply_button.grid(row=len(Commands), column=0, padx=10, pady=10, sticky="w")

        reject_button = ctk.CTkButton(edit_window, text="Reject", command=reject_changes)
        reject_button.grid(row=len(Commands), column=1, padx=10, pady=10, sticky="e")

        # 新增按钮，点击后打开显示 text 内容的窗口
        def open_text_window():
            """创建并展示显示 text 内容的新窗口"""
            text_window = ctk.CTkToplevel(edit_window)
            text_window.title("Text 内容")
            text_window.geometry("600x400")
            # 配置列和行的扩展性
            text_window.grid_columnconfigure(0, weight=1, uniform="equal")
            text_window.grid_rowconfigure(1, weight=1)  # 让文本框所在的行可以扩展
            
            # 创建文本框并展示 text 内容
            text_label = ctk.CTkLabel(text_window, text="显示 Text 内容:")
            text_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

            text_display = ctk.CTkTextbox(text_window, height=10, width=70)
            text_display.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
            text_display.insert("end", text)  # 显示传入的 text 内容
            text_display.configure(state="disabled")  # 禁用编辑

            # 关闭按钮
            close_button = ctk.CTkButton(text_window, text="Close", command=text_window.destroy)
            close_button.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        # 添加打开 `text` 内容窗口的按钮
        open_text_button = ctk.CTkButton(edit_window, text="查看 Text 内容", command=open_text_window)
        open_text_button.grid(row=len(Commands) + 1, column=0, columnspan=2, pady=10)
    
    def display_help(self):
        lines = ["输入cmdset，隔一个空格后输入：\n",
                 "help-查询指令\n",
                 "history-开启记录模式\n",
                 "nohistory-关闭记录模式\n",
                 "historyshow-查看记录\n",
                 "mode-转换模式\n",
                 "light-转换主题\n",
                 "model1-转换模型ERNIE-4.0-8K-Latest\n",
                 "model2-转换模型ERNIE-4.0-8K\n",
                 "model3-转换模型ERNIE-4.0-8K-Preview\n",
                 "model4-转换模型ERNIE-Speed-128K\n",
                 "model5-转换模型ERNIE-Speed-8K\n"]
        for line in lines:
            self.display_output(line)
    def change_ai_mode(self):
        if self.mode == "English":
            self.mode = "Chinese"
        elif self.mode == "Chinese":
            self.mode = "English"
        self.update_mode_label(self.mode)
if __name__ == "__main__":
    app = Cmd()  # 创建应用实例
    app.run()  # 运行主循环
