from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self):
        self.mnist = fetch_openml("mnist_784")
        self.data_frame = pd.DataFrame(self.mnist.data)
        plt.figure(figsize=(20, 4))

    def render_visualization_data(self):
        for index, digit in zip(range(1, 9), self.mnist.data[:8].values):
            plt.subplot(1, 8, index)
            plt.imshow(np.reshape(digit, (28, 28)), cmap="binary_r")
            plt.title(f'Example: {str(index)}')

        plt.show()

    def render_errors_from_prediction(self, errors, input_test, output_test):
        for i, img_error in zip(range(1, 9), errors[8:16]):
            plt.subplot(1, 8, i)
            plt.imshow(np.reshape(output_test[img_error], (28, 28)), cmap="binary_r")
            plt.title(f"Original: {str(input_test[img_error])} Prediction: {str(output_test[img_error])}")
        plt.show()


if __name__ == "__main__":
    dataset = Dataset()
    dataset.render_visualization_data()