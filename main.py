#!/usr/bin/python
# vim: set fileencoding=iso-8859-15

import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
import time


class Ising(object):
    """docstring for ."""
    def __init__(self, N,p,T,config='random'):
        self.N = N
        self.p = p
        if type(config) == 'numpy.ndarray':
            self.spins = config
        elif config == 'random':
            self.spins = np.random.choice(a=[-1,1], size=(N, N), p=[p, 1-p])

        self.spins = config
        self.T = T



    def getEnergyChange(self,pos):
        NNSpins = self.spins[(pos[0]+1)%self.N,pos[1]]+self.spins[pos[0],(pos[1]+1)%self.N]+self.spins[(pos[0]-1)%self.N,pos[1]]+self.spins[pos[0],(pos[1]-1)%self.N]
        dE = 2*self.spins[pos[0],pos[1]]*NNSpins
        return dE

    # def Gflip(self):
    #     print("hello")
    #     x = np.random.randint(0,self.N)
    #     y = np.random.randint(0,self.N)
    #     E = self.getEnergyChange([x,y])
    #     if E<=0:
    #         self.spins[x,y]*=-1
    #         return E
    #     elif np.random.uniform(0,1)< np.exp(-E/self.T):
    #         self.spins[x,y]*=-1
    #         return E
    #     else:
    #         return 0


    # def Kflip(self):
    #     x1,y1,y2,x2=0,0,0,0
    #     while not(x1==x2 and y1==y2) and not(self.spins[x1,y1]== self.spins[x2,y2]):
    #         x1 = np.random.randint(0,self.N)
    #         y1 = np.random.randint(0,self.N)
    #         x2 = np.random.randint(0,self.N)
    #         y2 = np.random.randint(0,self.N)
    #
    #     E1 = self.getEnergyChange([x1,y1])
    #     E2 = self.getEnergyChange([x2,y2])
    #     E = E1+E2
    #
    #     if (abs(x1-x2)==1 or abs(y1-y2)==1 or (x1+x2)%self.N == 1 or (y1+y2)%self.N==1):
    #         E = 4
    #
    #     if E<=0:
    #         self.spins[x1,y1] = self.spins[x2,y2]
    #         return E
    #     elif np.random.uniform(0,1)< np.exp(-E/self.T):
    #         self.spins[x2,y2] = self.spins[x1,y1]
    #         return E
    #     else:
    #         return 0
    def flip(self):
        pass


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
                E += self.spins[i,j]*(self.spins[(pos[0]+1)%self.N,pos[1]]+self.spins[pos[0],(pos[1]+1)%self.N]+self.spins[(pos[0]-1)%self.N,pos[1]]+self.spins[pos[0],(pos[1]-1)%self.N])
        return -1*E/4

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
            if i%5 == 0 and animate==True:
                #print(i)
                plt.imshow(self.spins)
                plt.pause(0.005)

        #if animate==True:
        plt.show()

        C = np.var(Es)/(self.N**2*self.T**2)
        X = np.var(Ms)/(self.N**2*self.T)
        E = np.mean(Es)
        M=  np.mean(Ms)
        return [C,X,E,M,Et,Mt,E2,M2]



    def totalM(self):
        return np.sum(self.spins)

    def __str__(self):
        p =np.empty([np.shape(self.spins)[0],np.shape(self.spins)[1]],dtype=np.str)
        p[self.spins==-1]="1"
        p[self.spins==1]="-1"
        return(np.array2string(p))

    @classmethod
    def experiment(cls,N,p,n):

        temps = np.linspace(1.5,3,n)
        Cs = np.zeros(n)
        Xs = np.zeros(n)
        Es = np.zeros(n)
        Ms = np.zeros(n)
        Et = 0
        Mt = 0
        for i in range(0,n):
            # if method == 'G' or 'Glauber':
            #     I = Glauber(N,p,temps[i])
            # elif method == 'K' or 'Kawasaki':
            #     I = Kawasaki(N,p,temps[i])
            I = cls(N,p,temps[i])
            C,X,E,M,Et,Mt,E2,M2 = I.simulate(10000)
            Es[i] = E
            Ms[i] = M
            Cs[i] = C#E2-Et*2
            Xs[i] = X#M2-Mt*2
        plt.plot(Ms)

        np.savetxt('heatcapacity.csv',Cs,delimiter=',')
        np.savetxt('energy.csv',Es,delimiter=',')
        np.savetxt('temperatures.csv',temps,delimiter=',')
        np.savetxt('mag.csv',Ms,delimiter=',')
        np.savetxt('magsep.csv',Xs,delimiter=',')

    @classmethod
    def plots(cls):
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
        plt.savefig('{}.png'.format(cls))

class Glauber(Ising):
    def __init__(self,N,p,T,config='default'):
        if config =='default':
            config = np.ones(N*N).reshape((N,N))
        else:
            config='random'
        Ising.__init__(self,N,p,T,config)
    def flip(self):
        x = np.random.randint(0,self.N)
        y = np.random.randint(0,self.N)
        E = self.getEnergyChange([x,y])
        if E<0:
            self.spins[x,y]*=-1
            return E
        elif np.random.uniform(0,1)< np.exp(-E/self.T):
            self.spins[x,y]*=-1
            return E
        else:
            return 0


class Kawasaki(Ising):
    def __init__(self,N,p,T,config='default'):
        if config =='default':
            config = np.arange(N*N).reshape((N,N))
            config[np.where(config<N*N/2)] = -1
            config[np.where(config>=N*N/2)] = +1
        else:
            config='random'

        Ising.__init__(self,N,p,T,config)

    def flip(self):

        x1 = np.random.randint(0,self.N)
        y1 = np.random.randint(0,self.N)
        x2 = np.random.randint(0,self.N)
        y2 = np.random.randint(0,self.N)

        if self.spins[x1,y1]== self.spins[x2,y2]:#checks if the spins are the same, returns 0 if so
            return 0

        else:#if spins not the same
            E1 = self.getEnergyChange([x1,y1])
            E2 = self.getEnergyChange([x2,y2])
            E = E1+E2

        S1 = self.spins[x1,y1]
        S2 = self.spins[x2,y2]

        if (abs(x1-x2)==1 or abs(y1-y2)==1 or (x1+x2)%self.N == 1 or (y1+y2)%self.N==1):
            E -= 1
        if E<0:
            self.spins[x1,y1] = S2
            self.spins[x2,y2] = S1
            return E
        elif np.random.uniform(0,1)< np.exp(-E/self.T):
            self.spins[x1,y1] = S2
            self.spins[x2,y2] = S1

            return E
        else:
            return 0
def main():
    #I = Glauber(10,0.5,1)
    #I = Kawasaki(50,0.5,10)
    #I.simulate(10000,True)

    #Kawasaki.experiment(50,0.5,20)
    #Kawasaki.plots()

    Glauber.experiment(10,0.5,20)
    Glauber.plots()


main()
