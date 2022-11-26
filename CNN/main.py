import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import *
from pre_deal_data import *


# 训练模型
def train_model(dataset, epochs, nums):
    for i in range(epochs):
        epoch_loss = 0  # 每轮总误差
        print("--------第{}轮训练开始--------".format(i+1))
        model.train()

        # 构建训练数据
        for j in range(1, nums+1):
            unit_loss = 0  # 每个单元总误差
            # 获取每一个单元的数据
            train_sub_dataset = dataset.get_group(j).to_numpy()
            train_sub_dataset = torch.Tensor(train_sub_dataset)
            train_sub_dataset_length = len(train_sub_dataset)
            train_sub_datasets = torch.zeros((train_sub_dataset_length, 30, 14))
            for k in range(train_sub_dataset_length):
                x_train_tensor = torch.zeros(30, 14)

                if k < 29:
                    x_train_tensor[x_train_tensor.shape[0]-k-1:] = train_sub_dataset[0:k+1, 2:-1]
                else:
                    x_train_tensor = train_sub_dataset[k-29:k+1, 2:-1]

                # 训练模型
                x_train_tensor = Variable(torch.Tensor(x_train_tensor))
                train_sub_datasets[k] = x_train_tensor

            train_sub_datasets = train_sub_datasets.reshape(train_sub_dataset_length, 1, 30, 14)
            y_train_sub_datasets = train_sub_dataset[:, -1:]
            y_train_sub_datasets = Variable(torch.Tensor(y_train_sub_datasets))
            x_train_sub_dataset_load = DataLoader(train_sub_datasets, batch_size=3)
            y_train_sub_datasets_load = DataLoader(y_train_sub_datasets, batch_size=3)
            for train_tensors_final in zip(x_train_sub_dataset_load, y_train_sub_datasets_load):

                output = model(train_tensors_final[0])
                loss = criterion(output, train_tensors_final[1])

                unit_loss += loss.item()
                loss.backward()

            # 每个单元都训练后进行梯度更新、清零
            optimizer.step()
            optimizer.zero_grad()

            print("--------第{}单元的总误差为：{}".format(j, unit_loss))

            epoch_loss += unit_loss

        print("--------第{}轮的总误差为：{}".format(i+1, epoch_loss))
        print("--------第{}轮结束--------".format(i+1))


# 测试模型
def test_model(dataset, epochs, nums, actual_rul):
    for i in range(epochs):
        total_rmse = 0
        total_predict_data = []
        print("--------第{}轮测试开始--------".format(i+1))
        model.eval()

        with torch.no_grad():
            for j in range(1, nums+1):
                unit_outputs = []
                test_sub_dataset = dataset.get_group(j).iloc[:, 2:]
                test_sub_dataset_length = len(test_sub_dataset)
                test_sub_dataset = Variable(torch.Tensor(test_sub_dataset.to_numpy()))
                for k in range(test_sub_dataset_length):
                    x_test_tensor = torch.zeros(30, 14)

                    if k < 29:
                        x_test_tensor[x_test_tensor.shape[0] - k - 1:] = test_sub_dataset[0:k + 1, :]
                    else:
                        x_test_tensor = test_sub_dataset[k - 29:k + 1, :]

                    x_test_tensors_final = x_test_tensor.reshape(
                        (1, 1, x_test_tensor.shape[0], x_test_tensor.shape[1]))
                    output = model.forward(x_test_tensors_final)
                    unit_outputs.append(output)

                predict_data = max(unit_outputs[-1].detach().numpy(), 0)
                total_predict_data.append(predict_data)
                total_rmse = np.add(np.power((predict_data - actual_rul.to_numpy()[j - 1]), 2), total_rmse)

            total_rmse = (np.sqrt(total_rmse / nums)).item()
            print("第{}轮测试均方根：{}".format(i + 1, total_rmse))

        print("--------第{}轮测试结束--------".format(i+1))

    return total_rmse, total_predict_data


# 结果可视化
def result_visualize(actual_target, predict_result):
    targets = actual_target.join(pd.DataFrame(predict_result))
    targets = targets.sort_values('RUL', ascending=False)
    actual_rul = targets.iloc[:, 0].to_numpy()
    predict_rul = targets.iloc[:, 1].to_numpy()

    plt.figure(figsize=(10, 6))  # plotting
    plt.axvline(x=num_test_units, c='r', linestyle='--')

    plt.plot(actual_rul, label='Actual RUL')
    plt.plot(predict_rul, label='Predicted RUL (RMSE = {})'.format(round(data_rmse, 3)))
    plt.title('Remaining Useful Life Prediction')
    plt.legend()
    plt.xlabel("Samples")
    plt.ylabel("Remaining Useful Life")
    plt.savefig('./visualize_result/{} RUL Prediction with CNN.png'.format(round(data_rmse, 3)))
    plt.show()


train_epochs = 250
test_epochs = 1
# 引入数据集1
train_dataset, test_dataset, actual_targets = load_dataset1()
input_size = train_dataset.get_group(1).shape[1] - 3
num_train_units = len(train_dataset.size())
num_test_units = len(test_dataset.size())

# 构建模型
model = CNN()
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
# 损失函数
criterion = nn.MSELoss()
# 优化器
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if __name__ == '__main__':
    # 训练模型
    train_model(train_dataset, train_epochs, num_train_units)

    # 测试模型
    data_rmse, predict_results = test_model(test_dataset, test_epochs, num_test_units, actual_targets)
    predict_results = np.array(predict_results).reshape((100, 1))
    print(predict_results)

    # 结果可视化
    result_visualize(actual_targets, predict_results)
