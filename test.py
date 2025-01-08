# 导入tk
from tkinter import *
import tkinter as tk
import tkinter.font as tf
from tkinter import ttk

def data_(file_name):
    if "cerevisiae" in file_name:
        lens = 31
        na = "cerevisiae"
        f_1 = open(".\cerevisiae_positive_314.txt", "r")
        f_2 = open(".\cerevisiae_negitive_314.txt", "r")
    elif "musculus" in file_name:
        lens = 21
        na = "musculus"
        f_1 = open(".\musculus_positive_472.txt", "r")
        f_2 = open(".\musculus_negitive_472.txt", "r")
    else:
        lens = 21
        na = "sapiens"
        f_1 = open(".\sapiens_positive_495.txt", "r")
        f_2 = open(".\sapiens_negitive_495.txt", "r")
    l, n = [], []
    for (text, text_2)  in zip(f_1.readlines(), f_2.readlines()):
        w = text.strip(">")
        l.append(">H."+na+"_" + w.strip("\n"))
        n.append(">H."+na+"_" + text_2.strip("\n").strip(">"))
    l = l + n
    file_names = l[0::2]
    site = ["Yes", "No"]
    # data = []
    # for i in file_names:
    #     if "P" == i:
    #         site = site[0]
    #     else:
    #         site = site[1]
    #     data.append([i, lens, site])

    data = [[i , lens, site[0]] if "P" in i  else [i , lens, site[1]] for i in file_names ]
    return data


def make_tk():
    # 创建主窗口
    window = tk.Tk()
    window.title('主窗口')
    window.geometry('1200x730+0+0')
    window.config(bg="white")
    var = tk.StringVar()
    border = LabelFrame(window, bg="black")
    border.pack(side="left")
    l = tk.Label(border, bg="white", width=30, height=70, text='Please select the data set you want：', font=("宋体", 19),fg="black", wraplength=300)
    l.pack(side="left")

    def data_show():
        text.delete("1.0", "end")
        font1 = tf.Font(family='微软雅黑', size=12)
        if "cerevisiae" in var.get():
            f_1 = open(".\cerevisiae_positive_314.txt", "r")
            f_2 = open(".\cerevisiae_negitive_314.txt", "r")
        elif "musculus" in var.get():
            f_1 = open(".\musculus_positive_472.txt", "r")
            f_2 = open(".\musculus_negitive_472.txt", "r")
        else:
            f_1 = open(".\sapiens_positive_495.txt", "r")
            f_2 = open(".\sapiens_negitive_495.txt", "r")
        for i in f_1.readlines():
            text.insert(END, i)
        for i in f_2.readlines():
            text.insert(END, i)
        text.config(font=font1, fg="black")
    # 按钮1及其功能
    r1 = tk.Radiobutton(window,command=data_show, text='cerevisiae', variable=var, value='cerevisiae',  width=27,height=1,fg='black',bg="white",font=('宋体',20))
    r1.pack(pady=10)

    r2 = tk.Radiobutton(window,command=data_show, text='musculus', variable=var, value='musculus' , width=27,height=1,fg='black',bg="white",font=('宋体',20))
    r2.pack(pady=10)

    r3 = tk.Radiobutton(window,command=data_show, text='sapiens', variable=var, value='sapiens',  width=27,height=1,fg='black',bg="white",font=('宋体',20))
    r3.pack(pady=10)
    r1.select()# 默认选择

    # 触发功能即按下按钮后想要程序做什么
    def print_selection():
        l.config(text='Is predicting ' + var.get() + " dataset sites for you.")
        if var.get() == "cerevisiae":
            NegativeCSV = r".\cerevisiae_negitive_314.txt"
            PositiveCSV = r".\cerevisiae_positive_314.txt"
            OutputDir = r"D:\RNA 假尿苷实验\王苹假尿苷实验\cerevisiae_result"
        elif var.get() == "musculus":
            NegativeCSV = r".\musculus_negitive_472.txt"
            PositiveCSV = r".\musculus_positive_472.txt"
            OutputDir = r"D:\RNA 假尿苷实验\王苹假尿苷实验\musculus_result"
            # messagebox.showinfo("第二步")
        else:
            NegativeCSV = r".\sapiens_negitive_495.txt"
            PositiveCSV = r".\sapiens_positive_495.txt"
            OutputDir = r"D:\RNA 假尿苷实验\王苹假尿苷实验\sapiens_result"
        # funciton(PositiveCSV, NegativeCSV, OutputDir, 10);

        '''
        表格
        '''
        # 创建tk窗口
        win1 = tk.Tk()
        win1.title('数据显示：')
        win1.geometry('950x650+20+20')
        data = data_(var.get())

        # frame容器放置表格
        frame01 = Frame(win1)
        frame01.place(x=0, y=0, width=950, height=620)
        # 加载滚动条
        scrollBar = Scrollbar(frame01)
        scrollBar.pack(side=RIGHT, fill=Y)
        # 准备表格
        tree = ttk.Treeview(frame01, columns=('name', 'len', 'site'), show="headings",
                        yscrollcommand=scrollBar.set, height=950)

        # tree = ttk.Treeview(win1, columns=('name', 'len', 'site'), show="headings",
        #                     displaycolumns="#all", height=27)
        tree.column("name", anchor="center", width=320)
        tree.column("len", anchor="center", width=320)
        tree.column("site", anchor="center", width=310)
        tree.heading('name', text="Sequence",)
        tree.heading('len', text="Number of nucleotides",)
        tree.heading('site', text="Site",)
        tree.tag_configure("evenColor", background="lightblue")  # 设置标签

        # 设置关联
        scrollBar.config(command=tree.yview)

        style = ttk.Style()
        style.configure("Treeview.Heading", font=("宋体", 50))

        for itm in data:
            tree.insert("", tk.END, values=itm, tags=("evenColor"))
        tree.pack(expand=1)

    # 确认按钮
    a = tk.PanedWindow(sashrelief=tk.SUNKEN, background="lightgray", width=200)
    a.pack()
    btn1 = tk.Button(a,text='Confirm',command=print_selection, width=40,height=1,fg='black',bg="white",font=('宋体',20))
    a.add(btn1)
    # 显示数据内容
    scr1 = tk.Scrollbar(window)
    scr1.pack(side='right', fill=tk.Y)  # 垂直滚动条
    text = tk.Text(window, width=80, height=10, bg="white")
    text.pack(side='left', fill=tk.BOTH, expand=True)
    text.config(yscrollcommand=scr1.set)  # 滚动设置互相绑定
    scr1.config(command=text.yview)  # 滚动设置互相绑定

    window.mainloop()

make_tk()