#!/usr/bin/python
# vim: set fileencoding=iso-8859-15

import numpy as np
from itertools import permutations
import matplotlib.pyplot as plt
import time
import math
import sys
import datetime


class Ising(object):
    """docstring for ."""
    def __init__(self, N,p,T,config='random'):
        self.N = N
        self.p = p

        if config == 'random':
            self.spins = np.random.choice(a=[-1,1], size=(N, N), p=[p, 1-p])
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
        return -1*E/4

    def equ(self,equN):
        for i in range(0,equN):
            self.sweep()

    def simulate(self,n,animate=False):
        self.equ(100)
        print(self.T)
        Es = np.zeros(n/10,dtype=float)
        #Es[0] = self.totalEnergy()
        Ms = np.zeros(n,dtype=float)
        #Ms[0] = self.totalM()
        for i in range(0,n):
            self.sweep()
            if i%10==0:
                Es[i/10] = self.totalEnergy()
                Ms[i/10] = self.totalM()
                # Et += Es[i/10]
                # Mt += Ms[i/10]
                # E2 += Es[i/10]**2
                # M2 += Ms[i/10]**2
                if animate==True:
                    #print(i)
                    plt.imshow(self.spins)
                    plt.pause(0.005)

        #if animate==True:
        plt.show()

        C = np.var(Es)/(self.N**2*self.T**2)
        X = np.var(Ms)/(self.N**2*self.T)
        E = np.mean(Es)
        M=  np.mean(Ms)
        values = [E,M,C,X]
        errors = [0,0,Ising.errors(Es,5),Ising.errors(Es,5)]

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
        C = np.genfromtxt('./data/heatcapacity.{}.csv'.format(cls.__name__), delimiter=",").reshape(2,-1)
        E = np.genfromtxt('./data/energy.{}.csv'.format(cls.__name__), delimiter=",").reshape(2,-1)
        TN = np.genfromtxt('./data/temperatures{}.csv'.format(cls.__name__).format(cls.__name__), delimiter=",")
        N=int(TN[-1])
        T = TN[0:np.size(TN)-1]
        M = np.genfromtxt('./data/mag.{}.csv'.format(cls.__name__), delimiter=",").reshape(2,-1)
        X = np.genfromtxt('./data/magsep.{}.csv'.format(cls.__name__), delimiter=",").reshape(2,-1)
        f = plt.figure(figsize=(18, 10))
        plt.title("{}*{} Ising Model using {} dynamics".format(N,N,cls.__name__))
        print(np.size(E[0,:]))
        print(np.size(T))
        sp =  f.add_subplot(2, 2, 1 );
        plt.scatter(T,E[0,:])
        plt.plot(T,E[0,:])

        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');

        sp =  f.add_subplot(2, 2, 2 );
        plt.scatter(T,C[0,:])
        plt.plot(T,C[0,:])
        plt.errorbar(T,C[0,:],yerr=C[1,:])
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Specific Heat ", fontsize=20);         plt.axis('tight');

        sp =  f.add_subplot(2, 2, 3 );
        plt.scatter(T,abs(M[0,:]),color='r')
        plt.plot(T,abs(M[0,:]),color='r')

        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Magnetization ", fontsize=20);         plt.axis('tight');

        sp =  f.add_subplot(2, 2, 4 );
        plt.scatter(T,abs(X[0,:]),color='r')
        plt.plot(T,abs(X[0,:]),color='r')
        plt.errorbar(T,X[0,:],yerr=X[1,:])
        plt.xlabel("Temperature (T)", fontsize=20);
        plt.ylabel("Susceptibility", fontsize=20);         plt.axis('tight');

        plt.show()
        f.savefig('{}.png'.format(cls.__name__))

    @staticmethod
    def errors(data,n):
        samps = np.zeros(np.size(data)).reshape(n,-1)
        for i in range(0,n):
            samps[i] = data[i::n]
        x = np.mean(np.var(samps,axis=1))
        x2 = np.mean(np.var(samps**2,axis=1))
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
def main(argv):
    #try:
    animate = argv[2]
    N = int(argv[3])
    Tn = int(argv[4])
    if argv[1] == 'G' or argv[1] =='Glauber' or argv[1] =='B':
        if animate=='True':
            I = Glauber(N,0.5,T,config='random')
            I.simulate(10000,True)
        else:
            Glauber.experiment(N,0.5,Tn)
            Glauber.plots()
    if argv[1] == 'K' or argv[1] == 'Kawasaki' or argv[1] =='B':
        if animate=='True':
            I = Kawasaki(N,0.5,T,config='random')
            I.simulate(10000,True)
        else:
            Kawasaki.experiment(N,0.5,Tn)
            Kawasaki.plots()
        # else:
        #     print("Input format: [dynamics: 'G', 'K'],[animate: True, False] ,[N: int], [T: float]")
        #     quit()

    #

main(sys.argv)

# def main():
#     I = Ising(10,0.5,10)
#     print(I.errors(np.arange(0,100),10))
#
# main()
