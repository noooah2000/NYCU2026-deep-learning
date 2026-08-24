from activations import Sigmoid, ReLU, NoActivation
from loss_function import MSE, BCE
from optimizers import SGD, Adam
from model import NeuralNetwork
from utils import generate_linear, generate_XOR_easy, show_result
import numpy as np

config = {"data_type": "XOR",
          "epochs": 100000,
          "learning_rate": 0.1, 
          "hidden_units": 4,
          "activation": "Sigmoid",
          "loss_function" : "MSE",
          "optimizer": "SGD"}

def train(model, loss_fn, optimizer, epochs, X, y, epochs_list, losses_list):
    print(f"-----training part-----")
    for epoch in range(epochs):
        y_pred = model.forward(X=X)
        model.backward(X=X, y_true=y)
        optimizer.step()

        if epoch % 1000 == 0:
            loss = loss_fn.forward(y_pred, y)
            epochs_list.append(epoch)
            losses_list.append(loss)

            if epoch % 5000 == 0:
                print(f"epoch{epoch:6d} : [training loss : {loss:1.5f}]")
    
def test(model, loss_fn, X, y):
    print(f"-----testing part-----")
    y_pred = model.forward(X=X)
    prediction = (np.round(y_pred)).astype(int)
    for i in range(max(0, len(X)-10), len(X)):
        print(f"point{i+1:3d} : [ground truth : {y[i][0]}] [prediction : {prediction[i][0]}]")
    
    loss = loss_fn.forward(y_pred, y)
    predictions = (np.round(y_pred)).astype(int)
    accuracy = np.mean(predictions == y) * 100
    print(f"testing loss: {loss:1.5f}, accuracy: {accuracy:3.2f}%")
    print(
    f"config : \n"
    f"data type-> {config['data_type']}, "
    f"epochs-> {config['epochs']}, "
    f"learning rate-> {config['learning_rate']}, \n"
    f"hidden units-> {config['hidden_units']}, "
    f"activation-> {config['activation']}, "
    f"loss_function-> {config['loss_function']}, "
    f"optimizer-> {config['optimizer']}"
)


def workflow(config):
    data_type = config["data_type"]
    epochs = config["epochs"]
    learning_rate = config["learning_rate"]
    hidden_units = config["hidden_units"]
    match config["activation"]:
        case "Sigmoid":
            activation = Sigmoid
        case "ReLU":
            activation = ReLU
        case "Linear":
            activation = NoActivation
    loss_fn = MSE() if config["loss_function"] == "MSE" else BCE()
    optimizer = SGD if config["optimizer"] == "SGD" else Adam

    np.random.seed(0)
    X, y = generate_linear() if data_type == "Linear" else generate_XOR_easy()

    np.random.seed(42)
    model = NeuralNetwork(input_size=2, 
                        hidden_size=hidden_units, 
                        output_size=1, 
                        loss_fn=loss_fn,
                        activation=activation())
    opt = optimizer(model=model, lr=learning_rate)

    epochs_list, losses_list = [], []

    train(model, loss_fn, opt, epochs, X, y, epochs_list, losses_list)
    test(model, loss_fn, X, y)
    show_result(X=X, y_true=y, y_pred=model.y_pred, epochs_list=epochs_list,losses_list=losses_list)

workflow(config)