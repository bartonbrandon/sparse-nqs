import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz
from functools import reduce
import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx


class ToricCode():
       
    def __init__(self, Lx, Ly, periodic_bc):
        """ Initialize the toric code Hamiltonian (on a torus)
        """
        self.Lx = Lx
        self.Ly = Ly
        self.periodic_bc = periodic_bc
        self.graph = nk.graph.Square(length=self.Lx, pbc=self.periodic_bc)
        self.N = self.graph.n_edges
        self.num_horizontal_edges = self.Lx * self.Ly
        self.num_vertical_edges = self.Lx * self.Ly
        self.hilbert = nk.hilbert.Spin(s=1/2, N=self.N)
        self.H = self.hamiltonian()
        self.observable_dict = self.create_observable_dict()


    def hamiltonian(self):
        """ Generate the Hamiltonian for the toric code
        """
        # Define the Hamiltonian
        H = 0

        star_ops = self.star_ops()
        plaquette_ops = self.plaquette_ops()
        H = -reduce(lambda x, y: x + y, star_ops + plaquette_ops)

        return H
    
    def edge_index(self, x, y, direction):
        """
        Returns the index of the edge at position (x, y) in a given direction.
        direction = 'horizontal' or 'vertical'
        """
        if direction == 'horizontal':
            return x * self.Ly + y
        elif direction == 'vertical':
            return self.num_horizontal_edges + x * self.Ly + y
        
    def star_ops(self):
        """
        Returns the star operators for the toric code.
        """
        star_operators = []

        for x in range(self.Lx):
            for y in range(self.Ly):
                sites = []
                
                # Horizontal edges
                sites.append(self.edge_index(x, y, 'horizontal'))
                sites.append(self.edge_index((x - 1) % self.Lx, y, 'horizontal'))
                
                # Vertical edges
                sites.append(self.edge_index(x, y, 'vertical'))
                sites.append(self.edge_index(x, (y - 1) % self.Ly, 'vertical'))
                star_op = sigmax(self.hilbert, sites[0])*sigmax(self.hilbert, sites[1])*sigmax(self.hilbert, sites[2])*sigmax(self.hilbert, sites[3])
                star_operators.append(star_op)
        
        return star_operators


    def plaquette_ops(self):
        """
        Returns the plaquette operators for the toric code.
        """
        plaquette_operators = []

        for x in range(self.Lx):
            for y in range(self.Ly):
                sites = []
                
                # Edges around the plaquette
                sites.append(self.edge_index(x, y, 'horizontal'))
                sites.append(self.edge_index(x, (y + 1) % self.Ly, 'horizontal'))
                sites.append(self.edge_index(x, y, 'vertical'))
                sites.append(self.edge_index((x + 1) % self.Lx, y, 'vertical'))

                plaquette_op = sigmaz(self.hilbert, sites[0])*sigmaz(self.hilbert, sites[1])*sigmaz(self.hilbert, sites[2])*sigmaz(self.hilbert, sites[3])
                plaquette_operators.append(plaquette_op)

        return plaquette_operators
    
    
        
    """ Observable functions """
    def create_observable_dict(self):
        """ Create a dictionary of observables for the 2D TFIM model
            to track during training
        """
        observable_dict = {}
        # Single site observables
        for j in range(self.N):
            observable_dict[f'X_{j}'] = self.X(j)
            observable_dict[f'Z_{j}'] = self.Z(j)
                
        return observable_dict

    def Z(self, i):
        Z = nk.operator.LocalOperator(hilbert=self.hilbert, dtype=complex)
        Z += nk.operator.spin.sigmaz(self.hilbert, i)
        return Z
    
    def X(self, i):
        X = nk.operator.LocalOperator(hilbert=self.hilbert, dtype=complex)
        X += nk.operator.spin.sigmax(self.hilbert, i)
        return X
    
