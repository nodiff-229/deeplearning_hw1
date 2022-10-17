from genericpath import exists

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.optim import lr_scheduler

import data
import models
import os
import argparse


## Note that: here we provide a basic solution for training and validation.
## You can directly change it if you find something wrong or not good enough.

def train_model(model, train_loader, valid_loader, criterion, optimizer, save_dir, num_epochs=20):
    def train(model, train_loader, optimizer, criterion):
        model.train(True)
        total_loss = 0.0
        total_correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()


            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)

        epoch_loss = total_loss / len(train_loader.dataset)
        epoch_acc = total_correct.float() / len(train_loader.dataset)
        return epoch_loss, epoch_acc.item()

    def valid(model, valid_loader, criterion):
        model.train(False)
        total_loss = 0.0
        total_correct = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += torch.sum(predictions == labels.data)
        epoch_loss = total_loss / len(valid_loader.dataset)
        epoch_acc = total_correct.float() / len(valid_loader.dataset)
        return epoch_loss, epoch_acc.item()

    best_acc = 0.0
    epoch_train_loss = []
    epoch_train_accuracy = []
    epoch_valid_loss = []
    epoch_valid_accuracy = []
    num_epochs_array = [i + 1 for i in range(num_epochs)]

    if not exists(save_dir):
        os.makedirs(save_dir)
    for epoch in range(num_epochs):
        print('epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        print("training: {:.4f}, {:.4f}".format(train_loss, train_acc))
        valid_loss, valid_acc = valid(model, valid_loader, criterion)
        print("validation: {:.4f}, {:.4f}".format(valid_loss, valid_acc))

        print("第%d轮的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step()

        epoch_train_loss.append(train_loss)
        epoch_train_accuracy.append(train_acc)
        epoch_valid_loss.append(valid_loss)
        epoch_valid_accuracy.append(valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_model = model
            torch.save(best_model, os.path.join(save_dir, 'best_model.pt'))

    # 绘制训练曲线图
    plt.figure()
    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.subplot(121)
    plt.xlabel('epochs')  # x轴标签
    plt.ylabel('loss')  # y轴标签
    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(num_epochs_array, epoch_train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.plot(num_epochs_array, epoch_valid_loss, linewidth=1, linestyle="solid", label="valid loss", color='black')
    plt.legend()
    plt.title('Loss curve')

    plt.subplot(122)
    plt.xlabel('epochs')  # x轴标签
    plt.ylabel('accuracy')  # y轴标签

    # 以x_train_acc为横坐标，y_train_acc为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 增加参数color='red',这是红色。
    plt.plot(num_epochs_array, epoch_train_accuracy, color='red', linewidth=1, linestyle="solid", label="train acc")
    plt.plot(num_epochs_array, epoch_valid_accuracy, color='orange', linewidth=1, linestyle="solid", label="valid acc")
    plt.legend()
    plt.title('Accuracy curve')

    plt.savefig("../result.png")

    print("最高精确度:", best_acc)



def draw_TSNE(valid_loader, device):
    model = torch.load("./checkpoints/best_model.pt", map_location=torch.device('cpu'))
    new_model = nn.Sequential(*list(model.children())[:9])
    new_model.to(device)
    new_model.eval()
    i = 0
    for inputs, labels in valid_loader:
        i = i + 1
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = new_model(inputs)
        # iris数据集中有4个指标，原始数据为4维
        # 使用 TSNE 降到2维
        outputs = outputs.cpu()
        outputs = outputs.detach().numpy()
        labels = labels.cpu().detach().numpy()
        outputs = (outputs.reshape(outputs.shape[0], outputs.shape[1] * outputs.shape[2] * outputs.shape[3]))
        if i == 1:
            sample = outputs
            target = labels
            continue
        sample = np.vstack((sample, outputs))
        target = np.vstack((target, labels))

    x_tsne = TSNE(n_components=2, random_state=33).fit_transform(sample)
    plt.figure().set_size_inches(10, 6)
    # 按照不同类别不同颜色显示降维之后的数据散点图
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=target)
    plt.show()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='hw1')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='dataset path')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/', help='model save path')
    args = parser.parse_args()

    ## about model
    num_classes = 10

    ## about data
    data_dir = args.data_dir  ## You need to specify the data_dir first
    input_size = 224
    batch_size = 36

    ## about training
    num_epochs = 100
    lr = 0.001

    ## model initialization
    model = models.model_A(num_classes=num_classes)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")
    model = model.to(device)

    ## data preparation
    train_loader, valid_loader = data.load_data(data_dir=data_dir, input_size=input_size, batch_size=batch_size)

    ## optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, 10, gamma=0.8, last_epoch=-1)
    ## loss function
    criterion = nn.CrossEntropyLoss()
    train_model(model, train_loader, valid_loader, criterion, optimizer, args.save_dir, num_epochs=num_epochs)
    # draw_TSNE(valid_loader, device)
