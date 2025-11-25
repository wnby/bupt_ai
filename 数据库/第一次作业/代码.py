import pymysql
import customtkinter
class MyTask(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.host = 'localhost'
        self.user = 'root'
        self.pas = 'xxx'
        self.name = 'testdemo1'
        self.modify_window = None
        label1 = customtkinter.CTkLabel(self, text=f"weekly task",
                                        anchor="w")
        label1.grid(row=0, column=1, padx=10, pady=5)
        # 创建一个 CTkScrollableFrame
        self.scrollable_frame1 = customtkinter.CTkScrollableFrame(self, width=250, height=340)
        self.scrollable_frame1.grid(row=1, column=1, padx=20, pady=20)
        self.text_content_entry1 = customtkinter.CTkEntry(self.modify_window)
        # 初始化列表
        self.text_content_entry = customtkinter.CTkEntry(self.modify_window,
                                                         placeholder_text="Enter text_content here")
        self.endday_entry = customtkinter.CTkEntry(self.modify_window, placeholder_text="Enter endday here")
        self.weekly_tasks = []
        self.state1 = 0
        self.title("数据库作业可视化")
        self.geometry("300x460+330+330")
        self.weekly_task_refresh()

    def weekly_task_initialize(self):
        self.weekly_tasks = []
        # 数据库文件名
        db_file = 'date.db'
        # 表名
        table_name = 'weeklytask'

        # 连接到SQLite数据库
        conn = pymysql.connect(host=self.host, user=self.user, password=self.pas, database=self.name)
        cursor = conn.cursor()

        # 创建表的SQL语句
        create_table_sql = f'''
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        text_content TEXT,
                        week INTEGER,
                        color TEXT,
                        checked INTEGER
                    );
                    '''
        # 查询表中所有记录
        query_select_all = "SELECT * FROM weeklytask"
        cursor.execute(create_table_sql)
        cursor.execute(query_select_all)

        # 获取所有结果
        rows = cursor.fetchall()
        # 执行创建表的SQL语句
        cursor.execute(create_table_sql)
        # 提交更改并关闭连接
        conn.commit()

        # 遍历查询结果并将数据存入相应的列表
        if rows:
            for row in rows:
                self.weekly_tasks.append({
                    'text_content': row[0],
                    'week': row[1],
                    'color': row[2]
                })
        conn.close()
    def weekly_task_refresh(self):
        print("weekly_task_refresh")
        self.weekly_task_initialize()
        self.scrollable_frame1.forget()
        self.scrollable_frame1 = customtkinter.CTkScrollableFrame(self, width=250, height=340)
        self.scrollable_frame1.grid(row=1, column=1, padx=20, pady=20)
        rownumber = 0
        if self.weekly_tasks:
            for task in self.weekly_tasks:
                print(1)
                button = customtkinter.CTkButton(self.scrollable_frame1,
                                                 command=lambda t=task['text_content'],w=task['week']:self.weekly_task_modify(t,w),
                                                 text="修改", width=60)
                button.grid(row=rownumber, column=0, padx=10, pady=5)
                button = customtkinter.CTkButton(self.scrollable_frame1,
                                                 command=lambda t=task['text_content']:self.weekly_task_delete(t),
                                                 text="删除", width=60)
                button.grid(row=rownumber, column=1, padx=10, pady=5)
                rownumber += 1
                label = customtkinter.CTkLabel(self.scrollable_frame1, text=task['text_content'], anchor="w")
                label.grid(row=rownumber, column=0, padx=10, pady=5)
                rownumber +=1
                label = customtkinter.CTkLabel(self.scrollable_frame1, text=f"weekday:", anchor="w")
                label.grid(row=rownumber, column=0, padx=10, pady=5)
                label = customtkinter.CTkLabel(self.scrollable_frame1,
                                               text=f"{task['week']}", anchor="w")
                label.grid(row=rownumber, column=1, padx=10, pady=5)
                rownumber += 1
                label = customtkinter.CTkLabel(self.scrollable_frame1,
                                               text=f"  ", anchor="w")
                label.grid(row=rownumber, column=1, padx=10, pady=5)
                rownumber += 1
        button = customtkinter.CTkButton(self.scrollable_frame1,
                                         command=lambda: self.weekly_task_create(),
                                         text="添加", width=60)
        button.grid(row=rownumber, column=0, padx=10, pady=5)
    def weekly_task_modify(self,text_content,week):
        if self.state1 == 0:
            self.state1 = 1
            # 重新显示窗口
            self.open_modify_window(text_content,week)


            print("weekly_task_modifytext_content",text_content)
            # 添加关闭按钮
            self.button = customtkinter.CTkButton(self.modify_window,
                                             command=lambda: self.weekly_task_modify_Confirm(text_content),
                                             text="确定", width=80)
            self.button.pack(pady=20)
    def weekly_task_modify_Confirm(self,oldtextcontent):
        textcontent = self.text_content_entry.get()
        week = self.endday_entry.get()
        color = self.text_content_entry1.get()
        if not week.isdigit():
            self.hide_modify_window()
            return 0
        # 连接到数据库
        conn = pymysql.connect(host=self.host, user=self.user, password=self.pas, database=self.name)
        cursor = conn.cursor()

        # 检查是否存在匹配的记录
        query_check = "SELECT COUNT(*) FROM weeklytask WHERE text_content = %s"
        cursor.execute(query_check, (textcontent,))
        exists = cursor.fetchone()[0]

        if not exists or textcontent==oldtextcontent:
            # 更新记录
            query_update = """
            UPDATE weeklytask
            SET week = %s,  text_content = %s,color = %s
            WHERE text_content = %s
            """
            cursor.execute(query_update, (week, textcontent,color, oldtextcontent))

            # 提交更改
            conn.commit()
            self.state1 = 0
            self.weekly_task_refresh()
            print("Record updated successfully.")
        else:
            print("No matching record found to update.")

        # 关闭数据库连接
        conn.close()
        # 隐藏窗口
        self.hide_modify_window()

    def weekly_task_create(self):
        if self.state1 == 0:
            self.state1 = 1

            self.open_modify_window("任务名","星期（数字）")

            # 添加关闭按钮
            self.button = customtkinter.CTkButton(self.modify_window,
                                                  command=lambda: self.weekly_task_create_Confirm(),
                                                  text="确定", width=80)
            self.button.pack(pady=20)

    def weekly_task_create_Confirm(self):
        textcontent = self.text_content_entry.get()
        week = self.endday_entry.get()
        color = self.text_content_entry1.get()
        if not week.isdigit():
            self.hide_modify_window()
            return 0
        # 连接到数据库
        conn = pymysql.connect(host=self.host, user=self.user, password=self.pas, database=self.name)
        cursor = conn.cursor()
        # 查询 bigtask 表中的 text_content 是否有匹配 textcontent 的项
        query = "SELECT EXISTS(SELECT 1 FROM weeklytask WHERE text_content = %s LIMIT 1)"
        cursor.execute(query, (textcontent,))
        match_found = cursor.fetchone()[0]
        conn.close()
        # 处理查询结果
        if not match_found:
            # 连接到数据库
            conn = pymysql.connect(host=self.host, user=self.user, password=self.pas, database=self.name)
            cursor = conn.cursor()

            # 插入新记录到 bigtask 表
            query_insert = """
            INSERT INTO weeklytask (text_content, week , color ,checked)
            VALUES (%s, %s, %s, %s)
            """
            # 插入的值
            values = (textcontent, week, color,0)
            cursor.execute(query_insert, values)

            # 提交更改
            conn.commit()

            self.weekly_task_refresh()
        else:
            print(f"重复 {textcontent}")
        # 隐藏窗口
        self.hide_modify_window()

    def weekly_task_delete(self,oldtextcontent):
        if self.state1 == 0:
            # 连接到数据库
            conn = pymysql.connect(host=self.host, user=self.user, password=self.pas, database=self.name)
            cursor = conn.cursor()

            # 删除 text_content 为 oldtextcontent 的记录
            query_delete_text_content = "DELETE FROM weeklytask WHERE text_content = %s"
            cursor.execute(query_delete_text_content, (oldtextcontent,))

            # 提交更改
            conn.commit()
            # 关闭数据库连接
            self.weekly_task_refresh()

    def open_modify_window(self,text_content,endday):
        if not self.modify_window or not self.modify_window.winfo_exists():
            self.modify_window = customtkinter.CTkToplevel(self)
            self.modify_window.geometry("300x250")
            # 初始化列表
            self.text_content_entry = customtkinter.CTkEntry(self.modify_window)
            self.endday_entry = customtkinter.CTkEntry(self.modify_window)
            # 创建一个新的窗口（弹出窗口）
            # 添加输入框，设置预文本
            self.text_content_entry.insert(0,text_content)

            self.endday_entry.insert(0,endday)
            self.text_content_entry.pack(pady=10)


            self.endday_entry.pack(pady=10)
            self.modify_window.title("弹出窗口")
            self.text_content_entry1 = customtkinter.CTkEntry(self.modify_window)

            self.text_content_entry1.insert(0, '0')
            self.text_content_entry1.pack(pady=10)
            # 绑定退出事件
            self.modify_window.protocol("WM_DELETE_WINDOW", self.on_close)


        else:
            # 如果窗口已经存在但被隐藏，则重新显示
            self.modify_window.deiconify()
    def on_close(self):
        # 在这里处理关闭窗口前需要执行的操作
        self.state1 = 0

        # 如果需要直接关闭窗口，调用 destroy() 方法
        self.modify_window.destroy()
    def hide_modify_window(self):
        if self.modify_window and self.modify_window.winfo_exists():
            self.state1 = 0
            # 隐藏窗口
            self.modify_window.destroy()
current_mode = customtkinter.get_appearance_mode()
customtkinter.set_appearance_mode(current_mode)
app = MyTask()
app.mainloop()