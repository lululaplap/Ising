import numpy as np
from itertools import permutations

class Ising(object):
    """docstring for ."""
    def __init__(self, N,p):
        self.N = N
        self.p = p
        self.spins = np.random.choice(a=[-1,1], size=(N, N), p=[p, 1-p])
        perms = np.array(list(permutations([1,0],2)))
        self.NN = np.array([perms,-1*perms])
        np.reshape(self.NN,(2,4))


    def getEnergyChange(self,pos):
        NNSpins = 0
        NN = self.NN+pos
        print()
        NNSpins = np.where(tuple(np.reshape(NN,(4,2)))]
        print(NNSpins)
        dE = 2*self.spins[pos[0],pos[1]]*np.sum(NNSpins)
        return dE
    def flip(self):
        x = np.random.randint(0,self.N)
        y = np.random.randint(0,self.N)

def main():
    for i in range(0,1):
        I = Ising(5,0.5)
        dE = I.getEnergyChange(np.array([2,2]))
        print(dE)

main()
