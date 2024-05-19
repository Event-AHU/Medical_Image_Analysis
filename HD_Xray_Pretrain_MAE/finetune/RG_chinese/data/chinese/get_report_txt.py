num_prelist = []
num_labellist = []
for i in range(0, 5000, 50):
    pre_head = f'第{i}个预测值'
    num_prelist.append(pre_head)
print(num_prelist)
for i in range(0, 5000, 50):
    label_head = f'第{i}个真实值'
    num_labellist.append(label_head)
print(num_labellist)

report_aug = []
# 打开文件
file_path = '/gpfs/home/22054/R2Gen-mae-224/results/mae_large_224_90_bs16_roberta_9223_ss50_20w_aug_test/reports.txt'  # 请替换为您的文件路径
with open(file_path, 'r') as file:
    # 读取文件内容
    for line in file.readlines():
        for x in num_prelist:
            if x in line:
                # print(line)
                report_aug.append(line)
                
report = []   
report_label = []         
# 打开文件
file_path2 = '/gpfs/home/22054/R2Gen-mae-224/results/mae_large_224_90_bs16_roberta_9223_ss50_20w/reports.txt'  # 请替换为您的文件路径
with open(file_path2, 'r') as file:
    # 读取文件内容
    for line2 in file.readlines():
        for x2 in num_prelist:
            if x2 in line2:
                # print(line2)
                report.append(line2)
        
        for x3 in num_labellist:
            if x3 in line2:
                # print(line2)
                report_label.append(line2)
                
# breakpoint()

with open('/gpfs/home/22054/R2Gen-mae-224/data/chinese/report_select.txt', 'w') as output_file:
    if len(report_aug)==len(report)==len(report_label):
        for i in range(len(report_aug)):
            pre = report[i]
            pre_aug = report_aug[i]
            label = report_label[i]
            output_file.write('原始：\n')
            output_file.write(pre)
            output_file.write('增强后：\n')
            output_file.write(pre_aug)
            output_file.write(label)
print("#####")