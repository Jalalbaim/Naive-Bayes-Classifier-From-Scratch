import numpy as np
import struct
import sys
import tqdm

def calculate_priors(y):
    num_classes = 10
    priors = np.zeros(num_classes)
    for c in range(num_classes):
        priors[c] = np.sum(y == c)
    priors /= len(y)
    return priors

class NaiveBayesClassifier:
    def __init__(self, mode='discrete'):
        self.mode = mode
        self.priors = None
        self.likelihoods = None
        self.means = None
        self.variances = None
        self.num_classes = 10
        self.num_bins = 32
        self.pseudocount = 1
    
    def fit(self, X, y):
        self.priors = calculate_priors(y)
        n_samples, n_features = X.shape
        if self.mode == 'discrete':
            self.likelihoods = np.zeros((self.num_classes, n_features, self.num_bins))
            # Bin pixel values into discrete categories
            X_binned = (X // 8).astype(int)

            for c in tqdm.tqdm(range(self.num_classes)):
                X_class = X_binned[y == c]
                for i in range(X.shape[1]):
                    for b in range(self.num_bins):
                        self.likelihoods[c, i, b] = np.sum(X_class[:, i] == b) + 1
                    self.likelihoods[c, i] /= len(X_class) + self.num_bins

        elif self.mode == 'continuous':
            self.means = np.zeros((self.num_classes, n_features))
            self.variances = np.zeros((self.num_classes, n_features))

            for c in tqdm.tqdm(range(self.num_classes)):
                X_class = X[y == c]
                self.means[c, :] = np.mean(X_class, axis=0)
                self.variances[c, :] = np.var(X_class, axis=0) + 1e-2  # Avoid division by zero

        else:
            raise ValueError("Mode must be 'discrete' or 'continuous'.")
    
    def predict(self, X):
        X_binned = (X // 8 ).astype(int)
        n_samples, n_features = X.shape
        predictions = []
        
        if self.mode == 'discrete':
            for i in range(n_samples):
                print(f"Image {i}:")
                posteriors = np.zeros(self.num_classes)
                for c in range(self.num_classes):
                    posterior = np.log(self.priors[c])
                    for j in range(n_features):
                        likelihood = self.likelihoods[c, j, X_binned[i, j]]
                        posterior += np.log(likelihood + 1e-10)  # Add small value to prevent log(0)
                    posteriors[c] = posterior
                
                # Posterior normalization as requested
                posteriors /= np.sum(posteriors)
                
                prediction = np.argmin(posteriors)
                for c in range(self.num_classes):
                    print(f"{c}: {posteriors[c]}")
                print(f"Prediction: {prediction}\n")
                predictions.append(prediction)

        elif self.mode == 'continuous':
            for i in range(n_samples):
                print(f"Image {i}:")
                posteriors = np.zeros(self.num_classes)
                for c in range(self.num_classes):
                    posterior = np.log(self.priors[c])
                    for j in range(n_features):
                        mean = self.means[c, j]
                        var = self.variances[c, j]
                        posterior += -0.5 * np.log(2 * np.pi * var) - ((X[i, j] - mean) ** 2) / (2 * var)
                    posteriors[c] = posterior

                # Posterior normalization as requested
                posteriors /= np.sum(posteriors)
                
                prediction = np.argmin(posteriors)
                for c in range(self.num_classes):
                    print(f"{c}: {posteriors[c]}")
                print(f"Prediction: {prediction}\n")
                predictions.append(prediction)
        
        return np.array(predictions)

    def print_imagination(self):
        print("Imagination of numbers in Bayesian classifier:")
        if self.mode == 'discrete':
            for c in range(self.num_classes):
                image = np.zeros((28, 28))
                for i in range(28):
                    for j in range(28):
                        bin_idx = np.argmax(self.likelihoods[c, i * 28 + j])
                        image[i, j] = 1 if bin_idx >= (self.num_bins // 2) else 0
                print(f"{c}:")
                for row in image:
                    print("".join(['1' if pixel == 1 else '0' for pixel in row]))
        else:
            for c in range(self.num_classes):
                image = np.zeros((28, 28))
                for i in range(28):
                    for j in range(28):
                        if self.means[c, i * 28 + j] > 128:
                            image[i, j] = 1
                        else:
                            image[i, j] = 0
                print(f"{c}:")
                for row in image:
                    print("".join(['1' if pixel == 1 else '0' for pixel in row]))
    
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    correct_predictions = np.sum(predictions == y_test)
    error_rate = 1 - (correct_predictions / len(y_test))
    
    print("Posterior (in log scale):")
    for i in range(len(X_test)):
        print(f"Image {i}: {predictions[i]}, Answer: {y_test[i]}")
    
    print(f"Error rate: {error_rate :.4f}")
    return error_rate


def read_idx_file(file_path):
    with open(file_path, 'rb') as f:
        print(f"Reading file: {file_path.split('/')[-1]}")
        magic_number = struct.unpack('>I', f.read(4))[0]

        num_items = struct.unpack('>I', f.read(4))[0]
        print(f"Number of Items: {num_items}")

        if 'images' in file_path:
            num_rows = struct.unpack('>I', f.read(4))[0]
            num_cols = struct.unpack('>I', f.read(4))[0]
            print(f"Image Dimensions: {num_rows}x{num_cols}")

            item_data = f.read()
            items = np.frombuffer(item_data, dtype=np.uint8)
            items = items.reshape(num_items, num_rows, num_cols)
        else:
            label_data = f.read()
            items = np.frombuffer(label_data, dtype=np.uint8)

        print(f"Items array shape: {items.shape}")
        print("-"*50)
        return items


def main():
    X_train = read_idx_file('train-images.idx3-ubyte_')
    y_train = read_idx_file('train-labels.idx1-ubyte_')
    X_test = read_idx_file('t10k-images.idx3-ubyte_')
    y_test = read_idx_file('t10k-labels.idx1-ubyte_')

    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    mode_input_int = int(input("Discrete 0 or continuous 1 : "))
    if mode_input_int == 0:
        mode_input = 'discrete'
    else:
        mode_input = 'continuous'
    model = NaiveBayesClassifier(mode=mode_input)
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    model.print_imagination()


if __name__ == "__main__":
    save = input("Wanna save the output to a file? (y/n) :")
    print("discrete 0 or continuous 1")
    if save == 'y':    
        with open('output.txt', 'w') as f:
            sys.stdout = f
            main()
    else:
        main()
    
    sys.stdout = sys.__stdout__ 
