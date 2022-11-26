import pandas as pd


# 给数据集加上标签名
def add_features_names():
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    return index_names, setting_names, sensor_names, col_names


# 给数据集加上RUL
def add_rul(data):
    # 获取每一单元的时间周期
    group_units = data.groupby(by='unit_nr')
    max_time_cycle = group_units['time_cycles'].max()

    # 将获取的时间周期并回
    data = data.merge(max_time_cycle.to_frame(name='max_time_cycles'), left_on='unit_nr', right_index=True)

    # 获取RUL
    data['RUL'] = data['max_time_cycles'] - data['time_cycles']

    # 删除max
    data = data.drop("max_time_cycles", axis=1)

    return data


# 加载数据集1
def load_dataset1():
    root = './CMAPSSData/'
    index_name, setting_name, sensor_name, cols = add_features_names()
    train_data = pd.read_csv(root+'train_FD001.txt', sep="\s+", header=None, names=cols)
    test_data = pd.read_csv(root+'test_FD001.txt', sep="\s+", header=None, names=cols)
    actual_rul_data = pd.read_csv(root+'RUL_FD001.txt', sep="\s+", header=None, names=['RUL'])
    # actual_rul_data['RUL'].clip(upper=100, inplace=True)

    # 无用数据：1、传感器操作：2、数据未变化
    drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_name + drop_sensors

    # 构建训练集
    # 删除
    train_data.drop(labels=drop_labels, axis=1, inplace=True)

    # 归一化处理，使用最大最小归一化
    sub_index = train_data.iloc[:, 0:2]
    data = train_data.iloc[:, 2:]
    data_norm = (data - data.min()) / (data.max() - data.min())
    # data_norm = (data - data.mean()) / (data.std())
    train_data_norm = pd.concat([sub_index, data_norm], axis=1)

    # 利用时间周期构建RUL
    new_train_data = add_rul(train_data_norm)
    # new_train_data['RUL'].clip(upper=100, inplace=True)

    # 将数据按单元分好
    sub_train_datasets = new_train_data.groupby(by="unit_nr")

    # 构建测试集
    test_data.drop(labels=drop_labels, axis=1, inplace=True)
    sub_index = test_data.iloc[:, 0:2]
    data = test_data.iloc[:, 2:]
    data_norm = (data - data.min()) / (data.max() - data.min())
    # data_norm = (data - data.mean()) / (data.std())
    test_data_norm = pd.concat([sub_index, data_norm], axis=1)
    sub_test_datasets = test_data_norm.groupby(by="unit_nr")

    return sub_train_datasets, sub_test_datasets, actual_rul_data


# 加载数据集2
def load_dataset2():
    root = './CMAPSSData/'
    train_data = pd.read_csv(root + 'train_FD002.txt', sep="\s+", header=None, names=add_features())
    test_data = pd.read_csv(root + 'test_FD002.txt', sep="\s+", header=None, names=add_features())
    print(train_data.info)
    print(test_data.info)


# 加载数据集3
def load_dataset3():
    root = './CMAPSSData/'
    train_data = pd.read_csv(root + 'train_FD003.txt', sep="\s+", header=None, names=add_features())
    test_data = pd.read_csv(root + 'test_FD003.txt', sep="\s+", header=None, names=add_features())
    print(train_data.info)
    print(test_data.info)


# 加载数据集4
def load_dataset4():
    root = './CMAPSSData/'
    train_data = pd.read_csv(root + 'train_FD004.txt', sep="\s+", header=None, names=add_features())
    test_data = pd.read_csv(root + 'test_FD004.txt', sep="\s+", header=None, names=add_features())
    print(train_data.info)
    print(test_data.info)
