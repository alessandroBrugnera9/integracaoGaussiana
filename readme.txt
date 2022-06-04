Chamando o arquivo main.py, todos os exemplos serão rodados apresentando o resultado para cada exemplo, explicanod o problema e apresentando a diferença da aproximação a depender do número de nós.

Ao iniciar o arquivo chama-se a função createCoefficientsArray que lê o arquivo dados.txt e ja exporta os nós e pesos para arrays numpy, para acelerar o processo de cálculo.

a funcao main chama cada exemplo que executa funcao doubleintegral.
a funcao doubleintegral esta documentando com comentarios explicando seu funcionamento, esta aceita integracoes inciando tanto por dydx e dxdy.
a funcao doubleintegral chama a funcao Integralgauss para a execucao de forma iterada das integrais duplas