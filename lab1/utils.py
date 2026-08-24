import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        # distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(X, y_true, y_pred, epochs_list, losses_list):
    plt.figure(figsize=(15, 5))
    plt.subplot(1,3,1)
    plt.title('Ground Truth', fontsize=18)
    for i in range(X.shape[0]):
        plt.plot(X[i][0], X[i][1], 'ro' if y_true[i] == 0 else 'bo')

    plt.subplot(1,3,2)
    plt.title('Predict Result', fontsize=18)
    for i in range(X.shape[0]):
        plt.plot(X[i][0], X[i][1], 'ro' if y_pred[i] < 0.5 else 'bo')

    plt.subplot(1,3,3)
    plt.title('Learning Curve', fontsize=18)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.plot(epochs_list, losses_list)

    plt.tight_layout()
    plt.show()