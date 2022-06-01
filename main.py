from typing import Callable
from unittest import result
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
    np.save('data/weights6.npy', weights)

    nodes = np.zeros(4)
    weights = np.zeros(4)

    for i in range(11, 15):
        numbers = lines[i].split("\t")
        nodes[i-11] = float64(numbers[0])
        weights[i-11] = float64(numbers[1])

    np.save('data/nodes8.npy', nodes)
    np.save('data/weights8.npy', weights)

    nodes = np.zeros(5)
    weights = np.zeros(5)

    for i in range(20, 25):
        numbers = lines[i].split("\t")
        nodes[i-20] = float64(numbers[0])
        weights[i-20] = float64(numbers[1])

    np.save('data/nodes10.npy', nodes)
    np.save('data/weights10.npy', weights)


def getCoefficients(n: int):
    nodes = np.load('data/nodes{}.npy'.format(n))
    weights = np.load('data/weights{}.npy'.format(n))

    return(nodes, weights)


def integrateGauss(n: int, a: float64, b: float64, x: float64, mathematicalFunction: Callable[[float64, float64], float64]) -> float64:
    """
    calculate gauss quadratire using n elments from a to b of the function provided

    :param int n: number of coefficients for Gauss Quadrature
    :param float64 a: lower limit from the integral
    :param float64 b: upper limit from the integral
    :param function mathematicalFunction: mathematical function of the function to be integrated
    """
    nodes, weights = getCoefficients(n)

    result = float64(0)
    # calculate parts of gauss quadrature and applyting compensation for different limits
    for i in range(len(nodes)):
        node = ((b-a)/2*nodes[i]
                + (b+a)/2)
        result += weights[i]*mathematicalFunction(
            x,
            node)
        node = (-(b-a)/2*nodes[i]
                + (b+a)/2)
        result += weights[i]*mathematicalFunction(
            x,
            node)

    result *= (b-a)/2

    print(result)
    return result


def doubleIntegral(n: int, a: float64, b: float64,  cx: Callable[[float64], float64], dx: Callable[[float64], float64], mathematicalFunction: Callable[[float64, float64], float64]) -> float64:
    nodes, weights = getCoefficients(n)

    result = float64(0)
    # calculate parts of gauss quadrature and applyting compensation for different limits
    for i in range(len(nodes)):
        node = ((b-a)/2*nodes[i]
                + (b+a)/2)
        result += weights[i]*integrateGauss(
            n,
            cx(node),
            dx(node),
            node,
            mathematicalFunction
        )
        node = (-(b-a)/2*nodes[i]
                + (b+a)/2)
        result += weights[i]*integrateGauss(
            n,
            cx(node),
            dx(node),
            node,
            mathematicalFunction
        )

    result *= (b-a)/2

    print(result)

    return result


def example1(x: float64) -> float64:
    fx = 2*x+1
    return fx


def test1(x: float64, y: float64) -> float64:
    fxy = 1-x-y
    return fxy


def main():
    # print(integrateGauss(6, 10, 20, example1))
    # print(integrateGauss(8, 10, 20, example1))
    # print(integrateGauss(10, 10, 20, example1))

    doubleIntegral(
        6,
        0,
        1,
        lambda x: 0,
        lambda x: 1-x,
        test1
    )


main()
