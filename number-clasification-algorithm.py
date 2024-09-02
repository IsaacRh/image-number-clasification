from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score


class NumberClassifier:
    def __init__(self, data, target):
        # divide the model for test and train
        self.INPUT_train, self.INPUT_test, self.OUTPUT_train, self.OUTPUT_test = train_test_split(data, target, test_size=0.3)
        print(len(self.OUTPUT_train))
        print(len(self.OUTPUT_test))
        self.layers_number = None # layer from perceptron
        self.outputs_number = None # outputs of perceptron
        self.params_number = None # params to do model

        self.clf = None

        self.output_predict = None
        self.score = None



    def training(self):
        from sklearn.neural_network import MLPClassifier

        """
        hidden layer: number of layers beetween OUTPUT AND INPUT layers
        activation function: the sigmoid function
        solver: the optimization weight, gradient descent.
        """
        clf = MLPClassifier(hidden_layer_sizes=(50,), activation="logistic", solver="sgd")
        clf.fit(self.INPUT_train, self.OUTPUT_train)
        self.layers_number = clf.n_layers_
        self.outputs_number = clf.n_outputs_
        self.params_number = clf.coefs_

        return clf

    def testing(self):
        self.output_predict = self.clf.predict(self.INPUT_test)
        self.score = f1_score(self.OUTPUT_test, self.output_predict, average="weighted")

    def get_errors_classification(self):
        index = 0
        errors = []
        for label, predict in zip(self.OUTPUT_test, self.output_predict):
            if label != predict:
                errors.append(index)
            index += 1

        return errors



