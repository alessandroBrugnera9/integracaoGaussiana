import numpy as np
from numpy import float64
from pyparsing import line


def createCoefficientsArray():
    f = open("dados.txt", "r")
    lines = f.readlines()
    nodes = np.zeros(3)
    weights = np.zeros(3)

    for i in range(3, 6):
        numbers = lines[i].split("\t")
        nodes[i-3] = float64(numbers[0])
        weights[i-3] = float64(numbers[1])

    np.save('data/nodes6.npy', nodes)
    np.save('data/weights6.npy', nodes)
    
    nodes = np.zeros(4)
    weights = np.zeros(4)

    for i in range(11, 15):
        numbers = lines[i].split("\t")
        nodes[i-11] = float64(numbers[0])
        weights[i-11] = float64(numbers[1])

    np.save('data/nodes8.npy', nodes)
    np.save('data/weights8.npy', nodes)

    nodes = np.zeros(5)
    weights = np.zeros(5)

    for i in range(20, 25):
        numbers = lines[i].split("\t")
        nodes[i-20] = float64(numbers[0])
        weights[i-20] = float64(numbers[1])

    np.save('data/nodes10.npy', nodes)
    np.save('data/weights10.npy', nodes)

    
def getCoefficients(n: int):
    nodes = np.load('data/nodes{}.npy'.format(n))
    weights = np.load('data/weights{}.npy'.format(n))

    return(nodes,weights)
def integrateGauss(n: int, mathematicalFunction) -> float64:
    nodes, weights = readCoefficients(n)

    return result


def main():
    print(getCoefficients(10))


main()
