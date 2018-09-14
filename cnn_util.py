import torch
import torch.nn as nn
import torch.optim as optim
from dataset_util import load_dataset, get_minibatches
from cnn_model import Net

import cv2
import numpy as np


def train(batch_size=32, num_epochs=2000):
    train_set_x, train_set_y, test_set_x, test_set_y, y_classes = load_dataset()
    tr_mini_batches = get_minibatches(train_set_x, train_set_y, batch_size)
    num_classes = len(y_classes)
    net = Net(num_classes)
    num_batches = len(tr_mini_batches)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.85)

    for e in range(num_epochs):
        data_loss = 0.0
        for idx, batch in enumerate(tr_mini_batches):
            mini_batch_x, mini_batch_y = batch
            tr_shape = mini_batch_x.shape
            x_tensor = torch.from_numpy(mini_batch_x.reshape((tr_shape[0], 3, 64, 64)))
            x_tensor = x_tensor.type('torch.FloatTensor')
            y_tensor = torch.from_numpy(mini_batch_y)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            output = net(x_tensor)

            # loss
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

            # print statistics
            data_loss += loss.item()
            print('[%d, %5d] loss: %.5f' % (e + 1, idx + 1, loss.item()))
        print('Average loss after epoch %d is : %.5f' % (e + 1, data_loss / num_batches))
        test(test_set_x, test_set_y, net)
    return net


def test(test_x, test_y, net):
    te_shape = test_x.shape
    x_tensor = torch.from_numpy(test_x.reshape((te_shape[0], 3, 64, 64)))
    x_tensor = x_tensor.type('torch.FloatTensor')
    y_tensor = torch.from_numpy(test_y)
    with torch.no_grad():
        predicted_outputs = net(x_tensor)
        _, predicted = torch.max(predicted_outputs.data, 1)
        correct = 0
        correct += (predicted == y_tensor).sum().item()
        total = y_tensor.size(0)
        print('Accuracy of the network on the %d test images: %d %%' % (total, 100 * correct / total))


if __name__ == '__main__':
    trained_net = train()
    torch.save(trained_net.state_dict(), './mytraining2.pth')

    # train_set_x, train_set_y, test_set_x, test_set_y, y_classes = load_dataset()
    # net = Net(len(y_classes))
    # correct = 0
    # net.load_state_dict(torch.load('./mytraining.pth'))
    # for index in range(len(test_set_y)):
    #     img = test_set_x[index]
    #     label = test_set_y[index]
    #     x_tensor = torch.from_numpy(img.reshape((1, 3, 64, 64)))
    #     x_tensor = x_tensor.type('torch.FloatTensor')
    #     y_tensor = torch.from_numpy(np.array(label))
    #     pred = net(x_tensor)
    #     _, pred_op = torch.max(pred.data, 1)
    #     print(pred_op, label)
    #     if(pred_op == y_tensor):
    #         correct += 1
    #
    #     # cv2.imshow("image", img)
    #     # cv2.waitKey(0)
    # print(correct)



