import netket as nk
import netket.experimental as nkx
import jax
import jax.numpy as jnp
import optax 

# Note: this class is only set up for the toric code model with N=18 spins.
class VMC:
    """
    Class for ground state optimization of neural network quantum states.
    """
    
    def __init__(
            self,
            operator,
            network,
            *,
            use_minSR: bool = False,
            learning_rate: float=0.01,
            diag_shift: float=0.1,
            use_diag_shift_schedule: bool=False,
            num_samples: int=1000,
            chunk_size: int=1000,
            num_discard: int=5,
            num_chains: int=16,
            is_holomorphic: bool=False,
            use_tc_sampler: bool=False,
            ):
        
        """
        Initialze the VMC class with the given parameters.

        Args:
        -----
            operator: netket.operator
                The operator to be optimized.
            network: flax nn.Module
                The neural network to be optimized.
            use_minSR: bool
                Whether to use the minimum SR preconditioner.
            learning_rate: float
                The learning rate for the optimizer.
            diag_shift: float
                The diagonal shift for the preconditioner.
            use_diag_shift_schedule: bool
                Whether to use a schedule for the diagonal shift.
            num_samples: int
                The number of samples to be used in the optimization.
            chunk_size: int
                The chunk size for the optimization.
            num_discard: int
                The number of samples to discard.
            num_chains: int
                The number of chains to be used in the optimization.
            is_holomorphic: bool
                Whether the network is holomorphic.
            use_tc_sampler: bool
                Whether to use the toric code sampler.


        Returns:
        --------
            None
        """
        self.operator = operator
        self.network = network
        self.num_samples = num_samples
        self.chunk_size = chunk_size
        self.num_discard = num_discard
        self.learning_rate = learning_rate
        self.use_minSR = use_minSR
        self.diag_shift = diag_shift
        self.use_diag_shift_schedule = use_diag_shift_schedule
        self.num_chains = num_chains
        self.is_holomorphic = is_holomorphic
        self.use_tc_sampler = use_tc_sampler

    def set_up(self):
        """
        Set up the VMC driver.

        Args:
        -----
            None

        Returns:
        --------
            driver: netket.VMC
                The VMC driver.
        """
        # Intialize the sampler
        if self.use_tc_sampler:
            rule_probabilites = [0.5, 0.5]
            multiple_rules = nk.sampler.rules.MultipleRules([SingleSpinFlip(), VertexFlip()], probabilities=rule_probabilites)
            self.sampler = nk.sampler.MetropolisSampler(self.operator.hilbert, multiple_rules, n_chains=self.num_chains, reset_chains=False)
        else: 
            self.sampler = nk.sampler.MetropolisLocal(hilbert=self.operator.hilbert, n_chains=self.num_chains)
        
        # Construct the SGD optimizer
        self.optimizer = nk.optimizer.Sgd(learning_rate=self.learning_rate)
        # Construct the SR preconditioner
        if self.use_diag_shift_schedule:
            diag_shift_schedule = optax.linear_schedule(init_value=self.diag_shift, end_value=self.diag_shift*0.1, transition_steps=1000)
            self.preconditioner = nk.optimizer.SR(diag_shift=diag_shift_schedule, holomorphic=self.is_holomorphic)
        else:
            self.preconditioner = nk.optimizer.SR(diag_shift=self.diag_shift, holomorphic=self.is_holomorphic)

        # The variational state
        vstate = nk.vqs.MCState(self.sampler, 
                                self.network, 
                                n_samples=self.num_samples, 
                                n_discard_per_chain=self.num_discard, 
                                chunk_size=self.chunk_size,
                                )
        
        # Variational monte carlo driver
        if self.use_minSR:
            driver = nkx.driver.VMC_SRt(hamiltonian=self.operator.H, 
                                                    optimizer=self.optimizer, 
                                                    diag_shift=self.diag_shift, 
                                                    variational_state=vstate)
        else:
            driver = nk.VMC(hamiltonian=self.operator.H, optimizer=self.optimizer, preconditioner=self.preconditioner, variational_state=vstate)
        

        return driver
    

""" Toric code sampler functions """
@nk.utils.struct.dataclass
class SingleSpinFlip(nk.sampler.rules.MetropolisRule):
    
    def transition(self, sampler, machine, parameters, state, key, σ):
        # Deduce the number of MCMC chains from input shape
        n_chains = σ.shape[0]
        
        # Load the Hilbert space of the sampler
        hilb = sampler.hilbert
        
        # Split the rng key into 3: one for the spin flip, one for the vertex flip, and one for indexing
        key_spin, key_indx = jax.random.split(key, 2)
        
        # ** Single Spin Flip **
        # Pick one random site on every chain
        indxs_spin = jax.random.randint(
            key_indx, shape=(n_chains, 1), minval=0, maxval=hilb.size
        )
        
        # Flip that spin
        σp, _ = nk.hilbert.random.flip_state(hilb, key_spin, σ, indxs_spin)
        
        return σp, None
    
@nk.utils.struct.dataclass
class VertexFlip(nk.sampler.rules.MetropolisRule):
    # Use default_factory for mutable default fields
    vertex_spins: jnp.ndarray = nk.utils.struct.field(
        default_factory=lambda: jnp.array([
            [0, 9, 8, 17],
            [9, 1, 10, 7],
            [10, 11, 2, 8],
            [0, 12, 3, 14],
            [1, 12, 13, 4],
            [2, 5, 13, 14],
            [3, 15, 6, 17],
            [4, 15, 16, 7],
            [5, 16, 17, 8],
        ])
    )
    
    def transition(self, sampler, machine, parameters, state, key, σ):
        # Deduce the number of MCMC chains from input shape
        n_chains = σ.shape[0]
        
        # Load the Hilbert space of the sampler
        hilb = sampler.hilbert
        
        # Split the rng key into 3: one for the spin flip, one for the vertex flip, and one for indexing
        key_vertex, key_indx = jax.random.split(key, 2)

        # ** Vertex Flip **
        # Pick one random vertex on every chain
        n_vertices = self.vertex_spins.shape[0]
        indxs_vertex = jax.random.randint(
            key_vertex, shape=(n_chains, 1), minval=0, maxval=n_vertices
        )
        
        # Flip the spins associated with the selected vertex
        spins_to_flip = self.vertex_spins[indxs_vertex[:, 0]]  # indexing to match shapes
        σp, _ = nk.hilbert.random.flip_state(hilb, key_vertex, σ, spins_to_flip)
        
        return σp, None
