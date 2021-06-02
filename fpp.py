import numpy as np
import matplotlib.pyplot as plt
# from scipy import wiener
from scipy import signal


def readFile(path:str):
    return np.loadtxt(path)


def plot(signal1,label1,title):
    n = np.array(range(len(signal1)))
    plt.figure(title)
    plt.xlim(0,len(n))
    plt.plot(n,signal1,'r',label1)
    plt.legend()

def plotOverlapper(signal1,label1,signal2,label2,title):
    n = np.array(range(len(signal1)))
    plt.figure(title)
    plt.xlim(0,len(n))
    plt.plot(n,signal1,'r',label1)
    plt.plot(n,signal2,'g',label2)
    plt.legend()


def diagonalIndecies(mat,k,val):
    p = []
    r,c = np.diag_indices_from(mat)
    if k < 0:
        r = r[-k:]
        c = c[:k]
    elif k > 0:
        r = r[:-k]
        c = c[k:]
    else:
        r = r
        c = c

    for i in range(len(r)):
        p.append( (r[i],c[i]) )
    for i in range(len(p)):
        mat[p[i]] = val

    return mat

def RYY_calculate(signals,shift):
    sigma = 0
    for i in range(1,(len(signals) - shift) -1):
        sigma += signals[i] * signals[i+shift]
    return (sigma/(len(signals) - shift))


def filter(signals,order,sigma,c):
    order += 1 
    a = np.empty([order,order])
    b = np.empty([order,1])
    Ryy = np.empty([order,1])


    for i in range(order):
        print(i-1)
        Ryy[i] = RYY_calculate(signals,i-1)

    b = Ryy[:order]
    b[0] = b[0] - sigma
    temp = Ryy[:order] 
    for  j in range(order):
        a = diagonalIndecies(a,j,temp[j])

    for  k in range(-order+1,0):
        a = diagonalIndecies(a,k,temp[-k])


    a = np.linalg.inv(a)
    c = np.linalg.inv(c)
    h = np.matmul(np.matmul(a,c),b)
    signals = signal.convolve(signals.reshape(signals.shape[0],1),h,"same")
    return signals.reshape((360,))


if __name__ == '__main__':
    
    order = 5
    var = 0.01

    c0 = -1 
    c1 = -0.75 
    c2 = -0.5
    c3 = -0.25

    c = np.array([[c0,c1,c2,c3,0,0],
                    [c1,c0+c2,c3,0,0,0],
                    [c2,c1+c3,c0,0,0,0],
                    [c3,c2,c1,c0,0,0],
                    [0,c3,c2,c1,c0,0],
                    [0,0,c3,c2,c1,c0]])

y = readFile("distorted_ECG.txt")
y -= np.mean(y)
x = readFile("Original_ECG.txt")
x -= np.mean(x)

filtered = filter(y,order,var,c)
n = np.array(range(len(y)))

plotOverlapper(filtered,"Filtered",-y,"Distorted","Filtered + Distorted")
plotOverlapper(filtered,"Filtered",x,"Original","Filtered + original")

mean2error = np.mean((x-filtered)**2)
plt.show()