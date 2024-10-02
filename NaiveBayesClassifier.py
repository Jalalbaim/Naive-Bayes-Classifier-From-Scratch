# HMW 2 : Machine Learning
# Author: BAIM Mohamed Jalal 
# Date: 10/10/2019

# Part 1 : Naive Bayes Classifier

import numpy as np 
import struct
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
import math

class NaiveBayesClassifier:
    def __init__(self, mode='discrete', num_bins=32, pseudocount=1):
        self.mode = mode
        self.num_bins = num_bins
        self.pseudocount = pseudocount
        self.num_classes = 10
        self.priors = None
        self.likelihoods = None
        self.means = None
        self.variances = None
        self.train_images = None
        self.train_labels = None
        self.num_rows = None
        self.num_cols = None

    def fit(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels
        num_train = len(train_images)
        self.num_rows, self.num_cols = train_images[0].shape
        num_pixels = self.num_rows * self.num_cols

        if self.mode == 'discrete':
            # Discrete Naive Bayes
            num_bins = self.num_bins
            class_counts = np.zeros(self.num_classes)
            pixel_bin_counts = np.zeros((self.num_classes, num_pixels, num_bins))

            # Training
            print("Training Naive Bayes classifier in discrete mode...")
            for i in range(num_train):
                label = train_labels[i]
                class_counts[label] += 1
                for row in range(self.num_rows):
                    for col in range(self.num_cols):
                        pixel_value = train_images[i][row][col]
                        bin_idx = int(pixel_value // (256 / num_bins))
                        pixel_bin_counts[label][row * self.num_cols + col][bin_idx] += 1

            self.priors = class_counts / num_train
            self.likelihoods = (pixel_bin_counts + self.pseudocount) / \
                (class_counts[:, None, None] + num_bins * self.pseudocount)
        elif self.mode == 'continuous':
            # Continuous Naive Bayes
            num_features = self.num_rows * self.num_cols
            train_images_flat = train_images.reshape(num_train, num_features)

            # Calculate priors
            self.priors = np.zeros(self.num_classes)
            for label in train_labels:
                self.priors[label] += 1
            self.priors /= num_train

            # Calculate means and variances
            self.means = np.zeros((self.num_classes, num_features))
            self.variances = np.zeros((self.num_classes, num_features))
            print("Training Naive Bayes classifier in continuous mode...")
            for c in range(self.num_classes):
                X_class = train_images_flat[train_labels == c]
                self.means[c, :] = np.mean(X_class, axis=0)
                self.variances[c, :] = np.var(X_class, axis=0) + 1e-2 # 1e-2 avoid division by zero
        else:
            raise ValueError("Mode must be 'discrete' or 'continuous'.")

    def predict(self, test_images):
        num_test = len(test_images)
        predictions = np.zeros(num_test, dtype=int)
        if self.mode == 'discrete':
            num_pixels = self.num_rows * self.num_cols
            num_bins = self.num_bins
            test_images_flat = test_images.reshape(num_test, num_pixels)
            print("Testing Naive Bayes classifier in discrete mode...")
            for i in range(num_test):
                posteriors = np.zeros(self.num_classes)
                for class_idx in range(self.num_classes):
                    log_posterior = math.log(self.priors[class_idx])
                    for pixel_idx in range(num_pixels):
                        pixel_value = test_images_flat[i][pixel_idx]
                        bin_idx = int(pixel_value // (256 / num_bins))
                        likelihood = self.likelihoods[class_idx][pixel_idx][bin_idx]
                        if likelihood > 0:
                            log_posterior += math.log(likelihood)
                        else:
                            log_posterior += math.log(1e-10) 
                    posteriors[class_idx] = log_posterior

                print("Postirior (in log scale):")
                for label in range(self.num_classes):
                    print(f"{label}: {posteriors[label]}")

                predicted_label = np.argmax(posteriors)
                predictions[i] = predicted_label
                print(f"Prediction: {predicted_label}")
        elif self.mode == 'continuous':
            num_features = self.num_rows * self.num_cols
            test_images_flat = test_images.reshape(num_test, num_features)
            print("Testing Naive Bayes classifier in continuous mode...")
            for i in range(num_test):
                print(f"Image {i}:")
                log_posteriors = np.zeros(self.num_classes)
                for c in range(self.num_classes):
                    log_posterior = math.log(self.priors[c])
                    for j in range(num_features):
                        mean = self.means[c, j]
                        var = self.variances[c, j]
                        log_likelihood = -0.5 * np.log(2 * np.pi * var) - \
                            ((test_images_flat[i, j] - mean) ** 2) / (2 * var)
                        log_posterior += log_likelihood
                    log_posteriors[c] = log_posterior
                    print(f"{c}: {log_posteriors[c]}")

                predicted_label = np.argmax(log_posteriors)
                predictions[i] = predicted_label
                print(f"Prediction: {predicted_label}")
        else:
            raise ValueError("Mode must be 'discrete' or 'continuous'.")

        return predictions

    def print_imagination(self):
        if self.mode == 'discrete':
            print("\nImagination of digits in Naive Bayes classifier (Discrete Mode):")
            for class_idx in range(self.num_classes):
                print(f"\nDigit {class_idx}:")
                for row in range(self.num_rows):
                    for col in range(self.num_cols):
                        pixel_idx = row * self.num_cols + col
                        # Calculate the most probable bin
                        most_probable_bin = np.argmax(self.likelihoods[class_idx][pixel_idx])
                        # Convert bin index to pixel value
                        pixel_value = (most_probable_bin + 0.5) * (256 / self.num_bins)
                        if pixel_value > 127:
                            print('1', end=' ')
                        else:
                            print('0', end=' ')
                    print()
        elif self.mode == 'continuous':
            print("\nClassifier's Imagination of Each Digit (Continuous Mode):")
            for c in range(self.num_classes):
                print(f"\nDigit {c}:")
                mean_image = self.means[c].reshape(self.num_rows, self.num_cols)
                for row in range(self.num_rows):
                    for col in range(self.num_cols):
                        if mean_image[row, col] > 127:
                            print('1', end=' ')
                        else:
                            print('0', end=' ')
                    print()
        else:
            raise ValueError("Mode must be 'discrete' or 'continuous'.")

    def calculate_error_rate(self, y_true, y_pred):
        errors = np.sum(y_true != y_pred)
        error_rate = errors / len(y_true)
        print(f"Error rate: {error_rate}")
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
    file_paths = ['./t10k-images.idx3-ubyte_', './t10k-labels.idx1-ubyte_', './train-images.idx3-ubyte_', './train-labels.idx1-ubyte_']
    file_names = ['test_images', 'test_labels', 'train_images', 'train_labels']

    data = {}
    for file_path, file_name in zip(file_paths, file_names):
        data[file_name] = read_idx_file(file_path)
    
    mode_input = input("Enter the mode (discrete or continuous): ")
    nb_classifier = NaiveBayesClassifier(mode=mode_input)
    nb_classifier.fit(data['train_images'], data['train_labels'])
    predictions = nb_classifier.predict(data['test_images'])
    nb_classifier.calculate_error_rate(data['test_labels'], predictions)
    nb_classifier.print_imagination()


if __name__ == '__main__':
    main()
