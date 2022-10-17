from matplotlib import pyplot as plt

epoch_train_loss = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
epoch_train_accuracy = [i for i in range(9)]
epoch_valid_loss = [i for i in range(9)]
epoch_valid_accuracy = [i for i in range(90, 99)]
num_epochs=9
num_epochs_array = [i+1 for i in range(num_epochs)]
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
plt.plot(num_epochs_array, epoch_valid_loss, linewidth=1, linestyle="solid", label="valid loss",color='black')
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


plt.savefig("../1.png")


