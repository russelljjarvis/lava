import time

# Import Process level primitives.
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.resources import CPU
from lava.magma.core.model.model import AbstractProcessModel

# Import parent classes for ProcessModels for Hierarchical Processes.
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel

# Import execution protocol.
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

# Import decorators.
from lava.magma.core.decorator import implements, tag, requires

from scipy.special import erf


from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

# Import monitoring Process.
from lava.proc.monitor.process import Monitor

from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF
from convert_params import convert_rate_to_lif_params

# Import bit accurate ProcessModels.
from lava.proc.dense.models import PyDenseModelBitAcc
from lava.proc.lif.models import PyLifModelBitAcc


from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg

# Import io processes.
from lava.proc import io

from lava.proc.dense.models import PyDenseModelFloat
from lava.proc.lif.models import PyLifModelFloat


class EINetwork(AbstractProcess):
    """Network of recurrently connected neurons.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        shape_exc = kwargs.pop("shape_exc", (1,))
        bias_exc = kwargs.pop("bias_exc", 1)
        shape_inh = kwargs.pop("shape_inh", (1,))
        bias_inh = kwargs.pop("bias_inh", 1)
        # Factor controlling strength of inhibitory synapses relative to excitatory synapses.
        self.g_factor = kwargs.pop("g_factor", 4)
        # Factor controlling response properties of network.
        # Larger q_factor implies longer lasting effect of provided input.
        self.q_factor = kwargs.pop("q_factor", 1)
        weights = kwargs.pop("weights")

        full_shape = shape_exc + shape_inh

        self.state = Var(shape=(full_shape,), init=0)
        # Variable for possible alternative state.
        self.state_alt = Var(shape=(full_shape,), init=0)
        # Biases provided to neurons.
        self.bias_exc = Var(shape=(shape_exc,), init=bias_exc)
        self.bias_inh = Var(shape=(shape_inh,), init=bias_inh)
        self.weights = Var(shape=(full_shape, full_shape), init=weights)

        # Ports for receiving input or sending output.
        self.inport = InPort(shape=(full_shape,))
        self.outport = OutPort(shape=(full_shape,))


@implements(proc=EINetwork, protocol=LoihiProtocol)
@tag(
    "rate_neurons"
)  # Tag allows for easy selection of ProcessModel in case multiple are defined.
@requires(CPU)
class RateEINetworkModel(PyLoihiProcessModel):

    outport: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)
    inport: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    state: np.ndarray = LavaPyType(np.ndarray, float)
    state_alt: np.ndarray = LavaPyType(np.ndarray, float)
    bias_exc: np.ndarray = LavaPyType(np.ndarray, float)
    bias_inh: np.ndarray = LavaPyType(np.ndarray, float)
    weights: np.ndarray = LavaPyType(np.ndarray, float)

    # @st.cache
    def __init__(self, proc_params):
        super().__init__(proc_params=proc_params)

        self.dr_exc = proc_params.get("dr_exc")
        self.dr_inh = proc_params.get("dr_inh")

        self.shape_exc = proc_params.get("shape_exc")
        self.shape_inh = proc_params.get("shape_inh")

        self.proc_params = proc_params

        self.got_decay = False
        self.got_bias = False
        self.weights_scaled = False

    def get_decay(self):
        """Construct decay factor.
        """
        dr_full = np.array(
            [self.dr_exc] * self.shape_exc + [self.dr_inh] * self.shape_inh
        )
        self.decay = 1 - dr_full

        self.got_decay = True

    def get_bias(self):
        """Construce biases.
        """
        self.bias_full = np.hstack([self.bias_exc, self.bias_inh])
        self.got_bias = False

    def scale_weights(self):
        """Scale the weights with integration time step.
        """

        self.weights[:, self.shape_exc :] *= self.dr_exc
        self.weights[:, : self.shape_exc] *= self.dr_inh
        self.proc_params.overwrite("weights", self.weights)

        self.weights_scaled = True

    def state_update(self, state):
        """Update network state according to:
            r[i + 1] = (1 - dr)r[i] + Wr[i]*r*dr + bias*dr
        """
        state_new = self.decay * state  # Decay the state.
        state_new += self.bias_full  # Add the bias.
        state_new += self.weights @ erf(state)  # Add the recurrent input.
        return state_new

    def run_spk(self):
        """The run function that performs the actual computation during
        execution orchestrated by a PyLoihiProcessModel using the
        LoihiProtocol.
        """

        if not self.got_decay:
            self.get_decay()

        if not self.got_bias:
            self.get_bias()

        if not self.weights_scaled:
            self.scale_weights()

        a_in = self.inport.recv()
        self.state = self.state_update(self.state) + a_in
        self.outport.send(self.state)


def generate_gaussian_weights(dim, num_neurons_exc, q_factor, g_factor):
    """Generate connectivity drawn from a Gaussian distribution with mean 0
    and std of (2 * q_factor) ** 2  / dim.
    If a excitatory neuron has a negative weight, we set it to 0 and similarly adapt
    positive weights for inhibitory neurons.
    W[i, j] is connection weight from pre-synaptic neuron j to post-synaptic neuron i.
    
    Paramerters
    -----------
    dim : int
        Dimensionality of network
    num_neurons_exc : int
        Number of excitatory neurons
    q_factor : float
        Factor determining response properties of network
    g_factor : float
        Factor determining inhibition-excitation balance
        
    Returns
    -------
    weights : np.ndarray
        E/I weight matrix
    """
    # Set scaled standard deviation of recurrent weights, J = q_factor**2 * 6 / full_shape.
    J = (2 * q_factor) ** 2 / dim
    weights = np.random.normal(0, J, (dim, dim))

    # Impose constraint that neurons can **either** be excitatory (positive weight)
    # **or** inhibitory (negative weight).
    exc_conns = np.full(weights.shape, True)
    exc_conns[
        :, num_neurons_exc:
    ] = False  # Set entries for inhibitory neurons to False.
    inh_conns = np.invert(exc_conns)

    mask_pos_weights = weights > 0
    mask_neg_weights = weights < 0

    # Set negative weights of exciatory neurons to zero and similarly for inhibitory neurons.
    # This induce sparsity in the connectivity.
    weights[mask_neg_weights * exc_conns] = 0
    weights[mask_pos_weights * inh_conns] = 0

    # We finally need to increase the inhibitory weights by a factor to control balance.
    weights[inh_conns] *= g_factor

    return weights


def first_model_to_cache(num_steps, dim):
    # Generate weights and store them in parameter dictionary.
    network_params_balanced["weights"] = generate_gaussian_weights(
        dim,
        num_neurons_exc,
        network_params_balanced["q_factor"],
        network_params_balanced["g_factor"],
    )

    st.markdown("Execution and Results")

    rcfg = Loihi1SimCfg(select_tag="rate_neurons")
    run_cond = RunSteps(num_steps=num_steps)

    # Instantiating network and IO processes.
    network_balanced = EINetwork(**network_params_balanced)
    state_monitor = Monitor()

    state_monitor.probe(target=network_balanced.state, num_steps=num_steps)

    # Run the network.
    network_balanced.run(run_cfg=rcfg, condition=run_cond)
    states_balanced = state_monitor.get_data()[network_balanced.name][
        network_balanced.state.name
    ]
    network_balanced.stop()
    return states_balanced


def auto_cov_fct(acts, max_lag=100, offset=200):
    """Auto-correlation function of parallel spike trains.
    
    Parameters
    ----------
    
    acts : np.ndarray shape (timesteps, num_neurons)
        Activity of neurons, a spike is indicated by a one    
    max_lag : int
        Maximal lag for compuation of auto-correlation function
        
    Returns:
    
    lags : np.ndarray
        lags for auto-correlation function
    auto_corr_fct : np.ndarray
        auto-correlation function
    """
    acts_local = acts.copy()[
        offset:-offset
    ]  # Disregard time steps at beginning and end.
    assert (
        max_lag < acts.shape[0]
    ), "Maximal lag must be smaller then total number of time points"
    num_neurons = acts_local.shape[1]
    acts_local -= np.mean(acts_local, axis=0)  # Perform temporal averaging.
    auto_corr_fct = np.zeros(2 * max_lag + 1)
    lags = np.linspace(-1 * max_lag, max_lag, 2 * max_lag + 1, dtype=int)

    for i, lag in enumerate(lags):
        shifted_acts_local = np.roll(acts_local, shift=lag, axis=0)
        auto_corrs = np.zeros(acts_local.shape[0])
        for j, act in enumerate(acts_local):
            auto_corrs[j] = (
                np.dot(
                    act - np.mean(act),
                    shifted_acts_local[j] - np.mean(shifted_acts_local[j]),
                )
                / num_neurons
            )
        auto_corr_fct[i] = np.mean(auto_corrs)

    return lags, auto_corr_fct


def second_model_to_cache(num_steps, dim):
    # Defining new, larger q_factor.
    q_factor = np.sqrt(dim / 6)

    # Changing the strenghts of the recurrent connections.
    network_params_critical = network_params_balanced.copy()
    network_params_critical["q_factor"] = q_factor
    network_params_critical["weights"] = generate_gaussian_weights(
        dim,
        num_neurons_exc,
        network_params_critical["q_factor"],
        network_params_critical["g_factor"],
    )

    # Configurations for execution.
    # num_steps = 1000
    rcfg = Loihi1SimCfg(select_tag="rate_neurons")
    run_cond = RunSteps(num_steps=num_steps)

    # Instantiating network and IO processes.
    network_critical = EINetwork(**network_params_critical)
    state_monitor = Monitor()

    state_monitor.probe(target=network_critical.state, num_steps=num_steps)

    # Run the network.
    network_critical.run(run_cfg=rcfg, condition=run_cond)
    states_critical = state_monitor.get_data()[network_critical.name][
        network_critical.state.name
    ]
    network_critical.stop()
    return states_critical


# states_critical = second_model_to_cache()


@implements(proc=EINetwork, protocol=LoihiProtocol)
@tag("lif_neurons")
class SubEINetworkModel(AbstractSubProcessModel):
    def __init__(self, proc):

        convert = proc.proc_params.get("convert", False)

        if convert:
            proc_params = proc.proc_params._parameters
            # Convert rate parameters to LIF parameters.
            # The mapping is based on:
            # A unified view on weakly correlated recurrent network, Grytskyy et al., 2013.
            lif_params = convert_rate_to_lif_params(**proc_params)

            for key, val in lif_params.items():
                try:
                    proc.proc_params.__setitem__(key, val)
                except KeyError:
                    if key == "weights":
                        # Weights need to be updated.
                        proc.proc_params._parameters[key] = val
                    else:
                        continue

        # Fetch values for excitatory neurons or set default.
        shape_exc = proc.proc_params.get("shape_exc")
        shape_inh = proc.proc_params.get("shape_inh")
        du_exc = proc.proc_params.get("du_exc")
        dv_exc = proc.proc_params.get("dv_exc")
        vth_exc = proc.proc_params.get("vth_exc")
        bias_mant_exc = proc.proc_params.get("bias_mant_exc")
        bias_exp_exc = proc.proc_params.get("bias_exp_exc", 0)

        # Fetch values for inhibitory neurons or set default.
        du_inh = proc.proc_params.get("du_inh")
        dv_inh = proc.proc_params.get("dv_inh")
        vth_inh = proc.proc_params.get("vth_inh")
        bias_mant_inh = proc.proc_params.get("bias_mant_inh")
        bias_exp_inh = proc.proc_params.get("bias_exp_inh", 0)

        # Create parameters for full network.
        du_full = np.array([du_exc] * shape_exc + [du_inh] * shape_inh)
        dv_full = np.array([dv_exc] * shape_exc + [dv_inh] * shape_inh)
        vth_full = np.array([vth_exc] * shape_exc + [vth_inh] * shape_inh)
        bias_mant_full = np.array(
            [bias_mant_exc] * shape_exc + [bias_mant_inh] * shape_inh
        )
        bias_exp_full = np.array(
            [bias_exp_exc] * shape_exc + [bias_exp_inh] * shape_inh
        )
        weights = proc.proc_params.get("weights")
        weight_exp = proc.proc_params.get("weight_exp", 0)

        full_shape = shape_exc + shape_inh

        # Instantiate LIF and Dense Lava Processes.
        self.lif = LIF(
            shape=(full_shape,),
            du=du_full,
            dv=dv_full,
            vth=vth_full,
            bias_mant=bias_mant_full,
            bias_exp=bias_exp_full,
        )

        self.dense = Dense(weights=weights, weight_exp=weight_exp)

        # Recurrently connect neurons to E/I Network.
        self.lif.s_out.connect(self.dense.s_in)
        self.dense.a_out.connect(self.lif.a_in)

        # Connect incoming activation to neurons and elicited spikes to ouport.
        proc.inport.connect(self.lif.a_in)
        self.lif.s_out.connect(proc.outport)

        # Alias v with state and u with state_alt.
        proc.vars.state.alias(self.lif.vars.v)
        proc.vars.state_alt.alias(self.lif.vars.u)


class CustomRunConfigFloat(Loihi1SimCfg):
    def select(self, proc, proc_models):
        # Customize run config to always use float model for io.sink.RingBuffer.
        if isinstance(proc, io.sink.RingBuffer):
            return io.sink.PyReceiveModelFloat
        if isinstance(proc, LIF):
            return PyLifModelFloat
        elif isinstance(proc, Dense):
            return PyDenseModelFloat
        else:
            return super().select(proc, proc_models)


def third_model_to_cache(network_params_balanced, num_steps):
    rcfg = CustomRunConfigFloat(
        select_tag="lif_neurons", select_sub_proc_model=True
    )
    run_cond = RunSteps(num_steps=num_steps)

    # Instantiating network and IO processes.
    lif_network_balanced = EINetwork(**network_params_balanced, convert=True)
    outport_plug = io.sink.RingBuffer(shape=shape, buffer=num_steps)

    # Instantiate Monitors to record the voltage and the current of the LIF neurons.
    monitor_v = Monitor()
    monitor_u = Monitor()

    lif_network_balanced.outport.connect(outport_plug.a_in)
    monitor_v.probe(target=lif_network_balanced.state, num_steps=num_steps)
    monitor_u.probe(target=lif_network_balanced.state_alt, num_steps=num_steps)

    lif_network_balanced.run(condition=run_cond, run_cfg=rcfg)

    # Fetching spiking activity.
    spks_balanced = outport_plug.data.get()
    data_v_balanced = monitor_v.get_data()[lif_network_balanced.name][
        lif_network_balanced.state.name
    ]
    data_u_balanced = monitor_u.get_data()[lif_network_balanced.name][
        lif_network_balanced.state_alt.name
    ]

    lif_network_balanced.stop()
    return (data_v_balanced, data_u_balanced, spks_balanced)


def fourth_model(network_params_critical, num_steps):

    rcfg = CustomRunConfigFloat(
        select_tag="lif_neurons", select_sub_proc_model=True
    )
    run_cond = RunSteps(num_steps=num_steps)

    # Creating new new network with changed weights.
    lif_network_critical = EINetwork(**network_params_critical, convert=True)
    outport_plug = io.sink.RingBuffer(shape=shape, buffer=num_steps)

    # Instantiate Monitors to record the voltage and the current of the LIF neurons
    monitor_v = Monitor()
    monitor_u = Monitor()

    lif_network_critical.outport.connect(outport_plug.a_in)
    monitor_v.probe(target=lif_network_critical.state, num_steps=num_steps)
    monitor_u.probe(target=lif_network_critical.state_alt, num_steps=num_steps)

    lif_network_critical.run(condition=run_cond, run_cfg=rcfg)

    # st.markdown("""Fetching spiking activity.""")
    spks_critical = outport_plug.data.get()
    data_v_critical = monitor_v.get_data()[lif_network_critical.name][
        lif_network_critical.state.name
    ]
    data_u_critical = monitor_u.get_data()[lif_network_critical.name][
        lif_network_critical.state_alt.name
    ]

    lif_network_critical.stop()
    return (
        spks_critical,
        data_u_critical,
        data_v_critical,
        lif_network_critical,
        lif_params_critical,
    )


def fifth_model_to_cache(
    num_steps, data_u_critical, data_v_critical, network_params_critical
):

    u_low = np.quantile(data_u_critical.flatten(), 0.2)
    u_high = np.quantile(data_u_critical.flatten(), 0.8)
    v_low = np.quantile(data_v_critical.flatten(), 0.2)
    v_high = np.quantile(data_v_critical.flatten(), 0.8)

    lif_params_critical = convert_rate_to_lif_params(**network_params_critical)
    weights = lif_params_critical["weights"]
    bias = lif_params_critical["bias_mant_exc"]

    params = {
        "vth": {
            "bits": 17,
            "signed": "u",
            "shift": np.array([6]),
            "val": np.array([1]),
        },
        "u": {
            "bits": 24,
            "signed": "s",
            "shift": np.array([0]),
            "val": np.array([u_low, u_high]),
        },
        "v": {
            "bits": 24,
            "signed": "s",
            "shift": np.array([0]),
            "val": np.array([v_low, v_high]),
        },
        "bias": {
            "bits": 13,
            "signed": "s",
            "shift": np.arange(0, 3, 1),
            "val": np.array([bias]),
        },
        "weights": {
            "bits": 8,
            "signed": "s",
            "shift": np.arange(6, 22, 1),
            "val": weights,
        },
    }

    mapped_params = float2fixed_lif_parameter(params)

    # st.markdown(
    #    """ Using the mapped parameters, we construct the fully-fledged parameter dictionary for the E/I network Process using the LIF SubProcessModel."""
    # )

    # Set up parameters for bit accurate model
    lif_params_critical_fixed = {
        "shape_exc": lif_params_critical["shape_exc"],
        "shape_inh": lif_params_critical["shape_inh"],
        "g_factor": lif_params_critical["g_factor"],
        "q_factor": lif_params_critical["q_factor"],
        "vth_exc": mapped_params["vth"],
        "vth_inh": mapped_params["vth"],
        "bias_mant_exc": mapped_params["bias_mant"],
        "bias_exp_exc": mapped_params["bias_exp"],
        "bias_mant_inh": mapped_params["bias_mant"],
        "bias_exp_inh": mapped_params["bias_exp"],
        "weights": mapped_params["weights"],
        "weight_exp": mapped_params["weight_exp"],
        "du_exc": scaling_funct_dudv(lif_params_critical["du_exc"]),
        "dv_exc": scaling_funct_dudv(lif_params_critical["dv_exc"]),
        "du_inh": scaling_funct_dudv(lif_params_critical["du_inh"]),
        "dv_inh": scaling_funct_dudv(lif_params_critical["dv_inh"]),
    }

    # st.markdown(
    #    """ Execution of bit accurate model
    # Configurations for execution.
    # """
    # )

    # num_steps = 1000
    run_cond = RunSteps(num_steps=num_steps)

    # Define custom Run Config for execution of bit accurate models.
    class CustomRunConfigFixed(Loihi1SimCfg):
        def select(self, proc, proc_models):
            # Customize run config to always use float model for io.sink.RingBuffer.
            if isinstance(proc, io.sink.RingBuffer):
                return io.sink.PyReceiveModelFloat
            if isinstance(proc, LIF):
                return PyLifModelBitAcc
            elif isinstance(proc, Dense):
                return PyDenseModelBitAcc
            else:
                return super().select(proc, proc_models)

    def do_run_0():
        rcfg = CustomRunConfigFixed(
            select_tag="lif_neurons", select_sub_proc_model=True
        )

        lif_network_critical_fixed = EINetwork(**lif_params_critical_fixed)
        outport_plug = io.sink.RingBuffer(shape=shape, buffer=num_steps)

        lif_network_critical_fixed.outport.connect(outport_plug.a_in)

        lif_network_critical_fixed.run(condition=run_cond, run_cfg=rcfg)

        # Fetching spiking activity.
        spks_critical_fixed = outport_plug.data.get()

        lif_network_critical_fixed.stop()
        return spks_critical_fixed

    spks_critical_fixed = do_run_0()
    return spks_critical_fixed


def get_params(dim):
    num_neurons_exc = int(dim * 0.8)
    num_neurons_inh = dim - num_neurons_exc

    # Single neuron paramters.
    params_exc = {"shape_exc": num_neurons_exc, "dr_exc": 0.01, "bias_exc": 0.1}

    params_inh = {"shape_inh": num_neurons_inh, "dr_inh": 0.01, "bias_inh": 0.1}

    # Inhibition-exciation balance for scaling inhibitory weights to maintain balance (4 times as many excitatory neurons).
    g_factor = 4.5

    # Factor controlling the response properties.
    q_factor = 1

    # Parameters Paramters for E/I network.
    network_params_balanced = {}

    network_params_balanced.update(params_exc)
    network_params_balanced.update(params_inh)
    network_params_balanced["g_factor"] = g_factor
    network_params_balanced["q_factor"] = q_factor

    network_params_balanced["weights"] = generate_gaussian_weights(
        dim,
        num_neurons_exc,
        network_params_balanced["q_factor"],
        network_params_balanced["g_factor"],
    )

    q_factor = np.sqrt(dim / 6)

    # Changing the strenghts of the recurrent connections.
    network_params_critical = network_params_balanced.copy()
    network_params_critical["q_factor"] = q_factor
    network_params_critical["weights"] = generate_gaussian_weights(
        dim,
        num_neurons_exc,
        network_params_critical["q_factor"],
        network_params_critical["g_factor"],
    )
    return network_params_balanced, network_params_critical
