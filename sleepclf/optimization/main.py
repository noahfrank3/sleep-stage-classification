import logging

from optimization import Optimizer

if __name__ == '__main__':
    batch_size = 24
    k = 5

    logging.basicConfig(level=logging.INFO)

    optimizer = Optimizer(batch_size, k)
    optimizer.optimize()
