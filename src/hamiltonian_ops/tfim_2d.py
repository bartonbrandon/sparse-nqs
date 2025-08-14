import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz
import jax.numpy as jnp
import numpy as np
import networkx as nx

class TFIM2D():
       
    def __init__(
            self, 
            Lx: int, 
            Ly: int, 
            kappa: float, 
            periodic_bc: bool = False
            ):
        
        """ Initialize the 2D transverse field Ising model Hamiltonian
        """
        self.Lx = Lx
        self.Ly = Ly
        self.N = int(self.Lx*self.Ly)
        self.J = 1.0
        self.kappa = kappa
        self.periodic_bc = periodic_bc
        self.central_spin = np.floor(self.N / 2).astype(int)
        self.graph = nk.graph.Square(length=self.Lx, pbc=self.periodic_bc)
        self.hilbert = nk.hilbert.Spin(s=1/2, N=self.N)
        self.H = self.hamiltonian()
        self.observable_dict = self.create_observable_dict()

    def hamiltonian(self):
        """ Generate the Hamiltonian for the 2D TFIM
        """
        # Define the Hamiltonian
        H = 0

        g = self.graph.to_networkx()
        edge_list = nx.convert_node_labels_to_integers(g).edges

        # Ising interaction terms
        for (i,j) in edge_list:
            H -= self.J*sigmaz(self.hilbert,i)*sigmaz(self.hilbert,j)

        # Transverse field terms
        H -= sum([self.kappa*sigmax(self.hilbert,i) for i in range(self.N)])
        
        return H
    

    def create_observable_dict(self):
        """ Create a dictionary of observables for the 2D TFIM model
            to track during training
        """
        observable_dict = {
        }
        # Single site observables
        for j in range(self.N):
            observable_dict[f'X_{j}'] = self.X(j)
            observable_dict[f'Z_{j}'] = self.Z(j)
        # Two site observables
        for i in range(self.N):
            for j in range(self.N):
                observable_dict[f'XX_{i}_{j}'] = self.XX(i, j)
                observable_dict[f'ZZ_{i}_{j}'] = self.ZZ(i, j)
                
                
        return observable_dict
        
    """ Observable functions """
    
    def magnetization(self):
        magnetization = nk.operator.LocalOperator(self.hilbert, dtype=complex)
        for i in range(self.N):
            magnetization += nk.operator.spin.sigmaz(self.hilbert, i) * (1/self.N)

        return magnetization
    
    def tf_magnetization(self):
        magnetization = nk.operator.LocalOperator(self.hilbert, dtype=complex)
        for i in range(self.N):
            magnetization += nk.operator.spin.sigmax(self.hilbert, i) * (1/self.N)

        return magnetization
    
    def squared_magnetization(self):
        magnetization = nk.operator.LocalOperator(self.hilbert, dtype=complex)
        
        for i in range(self.N):
            for j in range(self.N):
                magnetization += nk.operator.spin.sigmaz(self.hilbert, i) * nk.operator.spin.sigmaz(self.hilbert, j) * (1/self.N**2)

        return magnetization
    
    def squared_tf_magnetization(self):
        magnetization = nk.operator.LocalOperator(self.hilbert, dtype=complex)
        
        for i in range(self.N):
            for j in range(self.N):
                magnetization += nk.operator.spin.sigmax(self.hilbert, i) * nk.operator.spin.sigmax(self.hilbert, j) * (1/self.N**2)

        return magnetization
    
    def ZZ(self, i, j):
        ZZ = nk.operator.LocalOperator(hilbert=self.hilbert, dtype=complex)
        ZZ += nk.operator.spin.sigmaz(self.hilbert, i) * nk.operator.spin.sigmaz(self.hilbert, j)
        return ZZ
    
    def XX(self, i, j):
        XX = nk.operator.LocalOperator(hilbert=self.hilbert, dtype=complex)
        XX += nk.operator.spin.sigmax(self.hilbert, i) * nk.operator.spin.sigmax(self.hilbert, j)
        return XX

    def Z(self, i):
        Z = nk.operator.LocalOperator(hilbert=self.hilbert, dtype=complex)
        Z += nk.operator.spin.sigmaz(self.hilbert, i)
        return Z
    
    def X(self, i):
        X = nk.operator.LocalOperator(hilbert=self.hilbert, dtype=complex)
        X += nk.operator.spin.sigmax(self.hilbert, i)
        return X
    
