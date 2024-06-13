import random
mydata=[1,2,3,4,5,6,7,8,9,0]
mydata_y=[11,22,33,44,55,66,77,88,99,00]
# 假设mydata和mydata_y是已有的数据集和标签列表
# 首先将数据集和标签列表一起打乱
combined = list(zip(mydata, mydata_y))
random.shuffle(combined)
print(combined)
# 将打乱后的数据重新分配给mydata和mydata_y
mydata[:], mydata_y[:] = zip(*combined)
print(mydata,mydata_y)

# 计算数据长度
total_length = len(mydata)
train_size = int(total_length * 0.8)  # 训练集占总数据量的80%
test_size = total_length - train_size  # 测试集占总数据量的20%

# 分割数据
mydata_train = mydata[:train_size]
mydata_train_y = mydata_y[:train_size]

mydata_test = mydata[train_size:]
mydata_test_y = mydata_y[train_size:]

# 输出四个列表
print("Training data:", mydata_train)
print("Training labels:", mydata_train_y)
print("Testing data:", mydata_test)
print("Testing labels:", mydata_test_y)
