import netket as nk
import netket.experimental as nkx
import optax 

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
