#!/usr/bin/python
# vim: set fileencoding=iso-8859-15

import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import time
import math
import sys
import datetime
fig = plt.figure()
ax = fig.add_subplot(1,1,1)


class Ising(object):
    """docstring for ."""
    def __init__(self, N,p,T,config='random'):
        self.N = N
        self.p = p
        q = (1-p)/2
        if config == 'random':
            self.spins = np.random.choice(a=[-1,1], size=(N, N), p=[p,1-p])
        # if type(config) == 'numpy.ndarray':
        else:
            self.spins = config

        #self.spins = config
        self.T = T

    def getEnergyChange(self,pos):
        NNSpins = self.spins[(pos[0]+1)%self.N,pos[1]]+self.spins[pos[0],(pos[1]+1)%self.N]+self.spins[(pos[0]-1)%self.N,pos[1]]+self.spins[pos[0],(pos[1]-1)%self.N]
        dE = 2*self.spins[pos[0],pos[1]]*NNSpins
        return dE

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
        return -1*E/2

    def equ(self,equN):
        for i in range(0,equN):
            self.sweep()
    def aniUpdate(self,i):
        for i in range(10):
            self.sweep()
        ax.clear()
        ax.imshow(self.spins,cmap='cool')

    def simulate(self,n,animate=False):
        self.equ(100)
        print(self.T)
        Es = np.zeros(n/10,dtype=float)
        Es[0] = self.totalEnergy()
        Ms = np.zeros(n/10,dtype=float)
        Ms[0] = self.totalM()
        m=10
        for i in range(1,int(n/m)):
            for j in range(m):
                self.sweep()

            Es[i] = self.totalEnergy()
            Ms[i] = self.totalM()

        k1 = (self.N**2*self.T**2)
        k2 = (self.N**2*self.T)
        C = np.var(Es)/k1
        X = np.var(Ms)/k2
        E = np.mean(Es)
        M=  np.mean(Ms)
        values = [E,M,C,X]
        errors = [np.std(E)/np.size(E),np.std(abs(M))/np.size(M),Ising.errors(Es,5,k1),Ising.errors(Ms,5,k2)]

        return [values,errors]



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
        Cs = np.zeros((n,2))
        Xs = np.zeros((n,2))
        Es = np.zeros((n,2))
        Ms = np.zeros((n,2))
        for i in range(0,n):
            I = cls(N,p,temps[i])

            [values,errors] = I.simulate(10000)

            Es[i,0],Es[i,1] = values[0], errors[0]
            Ms[i,0],Ms[i,1] = values[1], errors[1]
            Cs[i,0],Cs[i,1] = values[2], errors[2]
            Xs[i,0],Xs[i,1] = values[3], errors[3]

        # t = datetime.datetime.now()
        # np.savetxt('heatcapacity.{}.{}.csv'.format(cls.__name__,t),Cs,delimiter=',')
        # np.savetxt('energy.{}.{}.csv'.format(cls.__name__,t),Es,delimiter=',')
        # np.savetxt('mag.{}.{}.csv'.format(cls.__name__,t),Ms,delimiter=',')
        # np.savetxt('magsep.{}.{}.csv'.format(cls.__name__,t),Xs,delimiter=',')

        t = ''#datetime.datetime.now()
        np.savetxt('./data/heatcapacity.{}.{}csv'.format(cls.__name__,t),Cs,delimiter=',')
        np.savetxt('./data/energy.{}.{}csv'.format(cls.__name__,t),Es,delimiter=',')
        np.savetxt('./data/mag.{}.{}csv'.format(cls.__name__,t),Ms,delimiter=',')
        np.savetxt('./data/magsep.{}.{}csv'.format(cls.__name__,t),Xs,delimiter=',')

        tempsN = np.append(temps,[N])


        np.savetxt('./data/temperatures{}.{}csv'.format(cls.__name__,t),tempsN,delimiter=',')

    @classmethod
    def plots(cls):
        C = np.genfromtxt('./data/heatcapacity.{}.csv'.format(cls.__name__), delimiter=",").reshape(-1,2)
        print(C)
        E = np.genfromtxt('./data/energy.{}.csv'.format(cls.__name__), delimiter=",").reshape(-1,2)
        TN = np.genfromtxt('./data/temperatures{}.csv'.format(cls.__name__).format(cls.__name__), delimiter=",")
        N=int(TN[-1])
        T = TN[0:np.size(TN)-1]
        M = np.genfromtxt('./data/mag.{}.csv'.format(cls.__name__), delimiter=",").reshape(-1,2)
        X = np.genfromtxt('./data/magsep.{}.csv'.format(cls.__name__), delimiter=",").reshape(-1,2)
        f = plt.figure(figsize=(18, 10))
        plt.title("{}*{} Ising Model using {} dynamics".format(N,N,cls.__name__))

        sp =  f.add_subplot(2, 2, 1 );
        plt.scatter(T,E[:,0])
        plt.plot(T,E[:,0])
        plt.errorbar(T,E[:,0],yerr=E[:,1],fmt='-b')
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

        sp =  f.add_subplot(2, 2, 2 );
        plt.scatter(T,C[:,0])
        plt.plot(T,C[:,0])
        plt.errorbar(T,C[:,0],yerr=C[:,1],fmt='-b')
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Specific Heat ", fontsize=20);         plt.axis('tight');

        sp =  f.add_subplot(2, 2, 3 );
        plt.scatter(T,abs(M[:,0]),color='r')
        plt.plot(T,abs(M[:,0]),color='r')
        plt.errorbar(T,abs(M[:,0]),yerr=M[:,1],fmt='-r')
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Magnetization ", fontsize=20);         plt.axis('tight');

        sp =  f.add_subplot(2, 2, 4 );
        plt.scatter(T,X[:,0])
        plt.plot(T,X[:,0])
        plt.errorbar(T,X[:,0],yerr=X[:,1],fmt='-r')
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Susceptibility", fontsize=20);         plt.axis('tight');

        plt.show()
        f.savefig('{}.png'.format(cls.__name__))

    @staticmethod
    def errors(data,n,k):
        samps = np.zeros(np.size(data)).reshape(n,-1)
        for i in range(0,n):
            samps[i] = data[i::n]
        x = np.mean(np.var(samps,axis=1)/k)
        x2 = np.mean((np.var(samps,axis=1)/k)**2)
        return(np.sqrt(x2-x**2))

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
            E += 4
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
def main(argv):
    #try:
    animate = argv[2]
    N = int(argv[3])
    Tn = float(argv[4])
    #p=float(argv[5])
    if argv[1] == 'G' or argv[1] =='Glauber' or argv[1] =='B':
        if animate=='True':
            I = Glauber(N,0.5,Tn,config='random')
            #I.simulate(1000,True)
            ani = animation.FuncAnimation(fig, I.aniUpdate)
            plt.show()
        else:
            Glauber.experiment(N,0.5,10)
            Glauber.plots()
    if argv[1] == 'K' or argv[1] == 'Kawasaki' or argv[1] =='B':
        if animate=='True':
            I = Kawasaki(N,0.5,Tn,config='random')
            ani = animation.FuncAnimation(fig, I.aniUpdate)
            plt.show()
        else:
            Kawasaki.experiment(N,0.5,10)
            Kawasaki.plots()
        # else:
        #     print("Input format: [dynamics: 'G', 'K'],[animate: True, False] ,[N: int], [T: float]")
        #     quit()

    #

main(sys.argv)
