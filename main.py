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


def integrateGauss(n: int, a: float64, b: float64, fixedVariable: float64, mathematicalFunction: Callable[[float64, float64], float64]) -> float64:
    """
    calculate gauss quadratire using n elments from a to b of the function provided

    :param int n: number of coefficients for Gauss Quadrature
    :param float64 a: lower limit from the integral
    :param float64 b: upper limit from the integral
    :param float64 fixedVariable: fixed variable when integrating, when calculating double integral iteratively
    :param function mathematicalFunction: mathematical function to be integrated, the 1st arg is the fixed, 2nd is the node
    """
    nodes, weights = getCoefficients(n)

    result = float64(0)
    # calculate parts of gauss quadrature and applyting compensation for different limits
    for i in range(len(nodes)):
        node = ((b-a)/2*nodes[i]
                + (b+a)/2)
        result += weights[i]*mathematicalFunction(
            fixedVariable,
            node)
        node = (-(b-a)/2*nodes[i]
                + (b+a)/2)
        result += weights[i]*mathematicalFunction(
            fixedVariable,
            node)

    result *= (b-a)/2

    return result


def doubleIntegral(n: int, a: float64, b: float64,  c: Callable[[float64], float64], d: Callable[[float64], float64],mathematicalFunction: Callable[[float64, float64], float64]) -> float64:
    """
    calculate double integral iteratively using gauss quadratire, using n elments
    from a to b of the external integral, and c to d of the internal integral (these lmits can be dependent on the external variable(node))

    :param int n: number of coefficients for Gauss Quadrature
    :param float64 a: lower limit from the external integral 
    :param float64 b: upper limit from the external ntegral
    :param function c: upper limit from the internal integral, should be callable even when constant
    :param function d: upper limit from the internal integral, should be callable even when constant
    :param float64 fixedVariable: fixed variable when integrating, when calculating double integral iteratively
    :param function mathematicalFunction: mathematical function to be integrated, the 1st arg is the fixed, 2nd is the node
    """
    nodes, weights = getCoefficients(n)

    result = float64(0)
    # calculate parts of gauss quadrature and applyting compensation for different limits
    for i in range(len(nodes)):
        node = ((b-a)/2*nodes[i]
                + (b+a)/2)
        result += weights[i]*integrateGauss(
            n,
            c(node),
            d(node),
            node,
            mathematicalFunction
        )
        node = (-(b-a)/2*nodes[i]
                + (b+a)/2)
        result += weights[i]*integrateGauss(
            n,
            c(node),
            d(node),
            node,
            mathematicalFunction
        )

    result *= (b-a)/2

    return result


def example1(x: float64) -> float64:
    fx = 2*x+1
    return fx


def example1(n: int) -> float64:
    return doubleIntegral(
        n,
        0,
        1,
        lambda x: 0,
        lambda x: 1-x,
        lambda x,y: (1-x-y)
    )


def example21(n: int) -> float64:
    return doubleIntegral(
        n,
        0,
        1,
        lambda x: 0,
        lambda x: 1-x**2,
        lambda x, y: 1
    )

def example22(n: int) -> float64:
    return doubleIntegral(
        n,
        0,
        1,
        lambda y: 0,
        lambda y: np.sqrt(1-y),
        lambda y, x: 1
    )


def example32(n: int) -> float64:
    return doubleIntegral(
        n,
        0.1,
        0.5,
        lambda x: x**3,
        lambda x: x**2,
        lambda x,y: np.exp(y/x),
    )


def main():
    print(example21(6))
    print(example21(8))
    print(example21(10))
    print(example32(6))
    print(example32(8))
    print(example32(10))
main()
