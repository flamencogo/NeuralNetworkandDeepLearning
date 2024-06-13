# 假设这两个列表是你的输入
list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']

# 创建一个包含list2元素的新列表，这里我们使用了列表推导式
duplicated_list2 = [item for item in list2 for _ in range(10)]

# 合并list1和duplicated_list2
merged_list = list1 + duplicated_list2

print(merged_list)
