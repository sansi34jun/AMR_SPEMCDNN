import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import tensorflow as tf
import mltools
import SPEMCDNN as mcl
import dataset2016
from tensorflow.keras.models import load_model

# Load data
opt_datasetpath = './RML2016.10a_dict.pkl'
opt_data = 0
(mods, snrs, lbl), (_, _), (_, _), (X_test, Y_test), (_, _, test_idx) = dataset2016.load_data(opt_datasetpath, opt_data)

# Expand dimensions to match model input
X_test = np.expand_dims(X_test, axis=3)

def predict():
    # Load the model
    model = mcl.SPEMCDNN(classes=11)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights('model.keras')
    
    # Define modulation class labels
    class_labels = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', '4-PAM', '16-QAM', '64-QAM', 'QPSK', 'WBFM']
    
    # Perform predictions
    test_Y_hat = model.predict(X_test, batch_size=800)
    
    # Calculate confusion matrix
    confnorm, _, _ = mltools.calculate_confusion_matrix(Y_test, test_Y_hat, class_labels)
    
    # Ensure the directory exists
    os.makedirs('SPEMCDNN2016a/figure', exist_ok=True)

    # Plot confusion matrix
    acc = {}
    acc_mod_snr = np.zeros((11, len(snrs)))
    i = 0
    for snr in snrs:
        # extract classes @ SNR
        test_SNRs = [lbl[x][1] for x in test_idx]

        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]

        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        confnorm_i, cor, ncor = mltools.calculate_confusion_matrix(test_Y_i, test_Y_i_hat, class_labels)
        acc[snr] = 1.0 * cor / (cor + ncor)
        result = cor / (cor + ncor)

        with open('SPEMCDNN2016a/figure/acc111.csv', 'a', newline='') as f0:
            write0 = csv.writer(f0)
            write0.writerow([result])
        mltools.plot_confusion_matrix(confnorm_i, labels=class_labels, title="ACC=%.2f%%" % (100.0 * acc[snr]), save_filename="SPEMCDNN2016a/figure/Confusion(SNR=%d)(ACC=%2f).png" % (snr, 100.0 * acc[snr]))

        acc_mod_snr[:, i] = np.round(np.diag(confnorm_i) / np.sum(confnorm_i, axis=1), 3)
        i += 1

    # Plot accuracy curve
    plt.figure(figsize=(10, 6))
    for i, mod in enumerate(class_labels):
        plt.plot(snrs, acc_mod_snr[i], label=mod)
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("Classification Accuracy for Each Modulation Type")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('SPEMCDNN2016a/figure/accuracy_per_modulation.png')

    # Save accuracy data to CSV
    with open('SPEMCDNN2016a/figure/accuracy_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['SNR'] + class_labels)
        for idx, snr in enumerate(snrs):
            writer.writerow([snr] + list(acc_mod_snr[:, idx]))

    mltools.plot_confusion_matrix(confnorm, labels=class_labels, save_filename='SPEMCDNN2016a/figure/sclstm-a_total_confusion.png')

if __name__ == "__main__":
    predict() 