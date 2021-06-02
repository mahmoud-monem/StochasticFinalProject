import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import scipy
from scipy import signal
from scipy.signal import wiener


def kth_diag_indices(matrix, k, value):
    points = []
    rows, cols = np.diag_indices_from(matrix)
    if k < 0:
        rows = rows[-k:]
        cols = cols[:k]
    elif k > 0:
        rows = rows[:-k]
        cols = cols[k:]
    else:
        rows = rows
        cols = cols

    for i in range(len(rows)):
        points.append((rows[i], cols[i]))

    for i in range(len(points)):
        matrix[points[i]] = value

    return matrix


def calculate_Ryy(signal, shift):
    sigma = 0
    for i in range(1, len(signal) - shift - 1):
        sigma += signal[i] + signal[i + shift]

    return sigma / (len(signal) - shift)


def signal_filteration(signals, c, sigma2v, filter_order):
    filter_order += 1
    a = np.empty([filter_order, filter_order])
    b = np.empty([filter_order, 1])
    Ryy = np.empty([filter_order, 1])
    for i in range(filter_order):
        print(i - 1)
        Ryy[i] = calculate_Ryy(signals, i - 1)

    b = Ryy[: filter_order]
    b[0] = b[0] - sigma2v
    tempo = Ryy[: filter_order]

    for k in range(filter_order):
        a = kth_diag_indices(a, k, tempo[k])
    for k in range(-filter_order + 1, 0):
        a = kth_diag_indices(a, k, tempo[k * -1])

    print('A: ', a)
    print('C: ', c)
    print('B: ', b)

    a = np.linalg.inv(a)
    c = np.linalg.inv(c)

    h = np.matmul(np.matmul(a, c), b)

    print('H: ', h)
    signals = signal.convolve(signals.reshape(signals.shape[0], 1), h, 'same')
    return signals.reshape((360, ))


def plot(sig1, sig2, label1: str, label2: str, title: str):
    n = np.array(range(len(sig1)))
    # plt.figure(title)
    plt.xlim(0, len(n))
    plt.subplot(4, 1, 2)
    plt.plot(n, sig1, label=label1)
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(n, sig2, label=label2)
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(n, sig1, 'g', label=label1)
    plt.plot(n, sig2, 'b', label=label2)
    # plt.savefig('output/original_filtered')

    plt.legend()


def main():
    filter_order = 5
    c0 = -1
    c1 = -0.75
    c2 = -0.5
    c3 = -0.25
    varV = 0.01

    c = np.array([[c0, c1, c2, c3, 0, 0], [c1, c0 + c2, c3, 0, 0, 0], [c2, c1 + c3, c0,
                 0, 0, 0], [c3, c2, c1, c0, 0, 0], [0, c3, c2, c1, c0, 0], [0, 0, c3, c2, c1, c0]])

    y = np.loadtxt('data/distorted_ECG.txt')
    y -= np.mean(y)
    x = np.loadtxt('data/Original_ECG.txt')
    x -= np.mean(x)

    n = np.array(range(len(x)))
    plt.subplot(4, 1, 1)
    plt.plot(n, x, label='Original')
    plt.legend()
    # plt.savefig('output/original')

    filtered_signal = signal_filteration(y, c, varV, filter_order)

    plot(filtered_signal, -y, 'Filtered Signal',
         'Distorted Signal', 'Filtered and Distorted Signal')

    plt.show()

    mean_square_err = np.mean(np.sqrt((x - filtered_signal)**2))
    print(mean_square_err)


if __name__ == '__main__':
    main()
