import csv

def print_last_line(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        last_row = None
        for row in reader:
            last_row = row
        if last_row:
            return float(last_row[4])  # 转换为浮点数

for strategy in ["sort", "reverse", "most_freq", "dyck1", "dyck2", "hist", "double_hist", "conll"]:
    results = ""
    for time in ["60","300","600","3600"]:
        for finetune in ["incremental_finetune"]:
            filename = f'./output/{strategy}{finetune}{time}/results.csv'
            try:
                value = print_last_line(filename)
                results += "&"+"{:.2f}".format(value * 100)   # 使用.format()格式化
            except:
                results +="&0.00"
    print(results)
