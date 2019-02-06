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
        x = np.random.randint(0,self.N)
        y = np.random.randint(0,self.N)
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
        for i in range(0,self.N**2):
            dE += self.flip()
        return dE

    def totalEnergy(self):
        E= 0
        for i in range(self.N):
            for j in range(self.N):
                pos = [i,j]
                E += -1*self.spins[i,j]*self.spins[(pos[0]+1)%self.N,pos[1]]+self.spins[pos[0],(pos[1]+1)%self.N]+self.spins[(pos[0]-1)%self.N,pos[1]]+self.spins[pos[0],(pos[1]-1)%self.N]
        return E/4

    def equ(self,equN):
        for i in range(0,equN):
            self.sweep()

    def simulate(self,n,animate=False):
        self.equ(100)
        print(self.T)
        Es = np.zeros(n,dtype=float)
        Es[0] = self.totalEnergy()
        Ms = np.zeros(n,dtype=float)
        Ms[0] = self.totalM()
        Et = 0
        Mt = 0
        E2 = 0
        M2 = 0
        for i in range(1,n):
            self.sweep()
            Es[i] = self.totalEnergy()
            Ms[i] = self.totalM()
            Et += Es[i]
            Mt += Ms[i]
            E2 += Es[i]**2
            M2 += Ms[i]**2
            if i%50 == 0 & animate==True:
                plt.imshow(self.spins)
                plt.pause(0.005)

        if animate==True:
            plt.show()
        C = np.var(Es)/(self.N**2*self.T**2)
        X = np.var(Ms)/(self.N**2*self.T)
        E = np.mean(Es)
        M=np.mean(Ms)
        return [C,X,E,M,Et,Mt,E2,M2]



    def totalM(self):
        return np.sum(self.spins)

    def __str__(self):
        p =np.empty([np.shape(self.spins)[0],np.shape(self.spins)[1]],dtype=np.str)
        #print(p)
        p[self.spins==-1]="1"
        p[self.spins==1]="0"
        return(np.array2string(p))

    @staticmethod
    def experiment(N,p,n):

        temps = np.linspace(1,3,n)
        Cs = np.zeros(n)
        Xs = np.zeros(n)
        Es = np.zeros(n)
        Ms = np.zeros(n)
        Et = 0
        Mt = 0
        for i in range(0,n):
            I = Ising(N,p,temps[i])
            [C,X,E,M,Et,Mt,E2,M2] = I.simulate(10000)

            Es[i] = E
            Ms[i] = M
            Cs[i] = C#E2-Et*2
            Xs[i] = X#M2-Mt*2

        np.savetxt('heatcapacity.csv',Cs,delimiter=',')
        np.savetxt('energy.csv',Es,delimiter=',')
        np.savetxt('temperatures.csv',temps,delimiter=',')
        np.savetxt('mag.csv',Ms,delimiter=',')
        np.savetxt('magsep.csv',Xs,delimiter=',')

    @staticmethod
    def plots():
        C = np.genfromtxt('heatcapacity.csv', delimiter=",")
        E = np.genfromtxt('energy.csv', delimiter=",")
        T = np.genfromtxt('temperatures.csv', delimiter=",")
        M = np.genfromtxt('mag.csv', delimiter=",")
        X = np.genfromtxt('magsep.csv', delimiter=",")
        f = plt.figure(figsize=(18, 10))

        sp =  f.add_subplot(2, 2, 1 );
        plt.scatter(T,E)
        plt.plot(T,E)
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

        sp =  f.add_subplot(2, 2, 2 );
        plt.scatter(T,C)
        plt.plot(T,C)
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Specific Heat ", fontsize=20);         plt.axis('tight');

        sp =  f.add_subplot(2, 2, 3 );
        plt.scatter(T,abs(M))
        plt.plot(T,abs(M))
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Magnetization ", fontsize=20);         plt.axis('tight');

        sp =  f.add_subplot(2, 2, 4 );
        plt.scatter(T,abs(X))
        plt.plot(T,abs(X))
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Susceptibility", fontsize=20);         plt.axis('tight');

        plt.show()



def main():
    I = Ising(10,0.5,1)
    Ising.experiment(20,0.99,20)
    Ising.plots()
    #for i in range(0,1):
        #I.simulate(10000,True)
#
main()
