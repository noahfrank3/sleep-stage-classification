import os

from dotenv import load_dotenv

from optimization import Optimizer

if __name__ == '__main__':
    batch_size = 1
    k = 5

    optimizer = Optimizer(batch_size, k)
    optimizer.optimize()
