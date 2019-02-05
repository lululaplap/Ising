#!/usr/bin/python
# vim: set fileencoding=iso-8859-15

import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt


class Ising(object):
    """docstring for ."""
    def __init__(self, N,p,T):
        self.N = N
        self.p = p
        self.spins = np.random.choice(a=[-1,1], size=(N, N), p=[p, 1-p])
        self.T = T

    def getEnergyChange(self,pos):
        NNSpins = self.spins[(pos[0]+1)%self.N,pos[1]]+self.spins[pos[0],(pos[1]+1)%self.N]+self.spins[(pos[0]-1)%self.N,pos[1]]+self.spins[pos[0],(pos[1]-1)%self.N]
        dE = 2*self.spins[pos[0],pos[1]]*NNSpins
        return dE

    def flip(self):
        x = np.random.randint(0,self.N-1)
        y = np.random.randint(0,self.N-1)
        E = self.getEnergyChange([x,y])
        if E<=0:
            self.spins[x,y]*=-1
            return E
        elif np.random.uniform(0,1)< np.exp(-E/self.T):
            self.spins[x,y]*=-1
            return E
        else:
            return 0

    def sweep(self):
        dE = 0
        for i in range(0,self.N):
            dE += self.flip()
        return dE
    def totalEnergy(self):
        E= 0
        for i in range(self.N-1):
            for j in range(self.N-1):
                pos = [i,j]
                E += -1*self.spins[i,j]*self.spins[(pos[0]+1)%self.N,pos[1]]+self.spins[pos[0],(pos[1]+1)%self.N]+self.spins[(pos[0]-1)%self.N,pos[1]]+self.spins[pos[0],(pos[1]-1)%self.N]
        return E

    def simulate(self,n):
        Es = np.zeros(n,dtype=float)
        Es[0] = self.totalEnergy()
        Ms = np.zeros(n,dtype=float)
        Ms[0] = self.totalM()
        for i in range(1,n):
            Es[i] = self.sweep()+Es[i-1]
            Ms[i] = self.totalM()
            if i%1000000 == 0:
                plt.imshow(self.spins)
                plt.show()

        plt.plot(Es)
        plt.show()
        plt.plot(Ms)
        plt.show()


    def totalM(self):
        return np.sum(self.spins)

    def __str__(self):
        p =np.empty([np.shape(self.spins)[0],np.shape(self.spins)[1]],dtype=np.str)
        #print(p)
        p[self.spins==-1]="1"
        p[self.spins==1]="0"
        return(np.array2string(p))

def main():
    I = Ising(100,0.5,10000)
    for i in range(0,1):
        I.simulate(1000)

main()
