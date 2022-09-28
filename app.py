import time

# Import Process level primitives.
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.resources import CPU
from lava.magma.core.model.model import AbstractProcessModel

import pandas as pd
import pickle

# Import parent classes for ProcessModels for Hierarchical Processes.
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel

# Import execution protocol.
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

# Import decorators.
from lava.magma.core.decorator import implements, tag, requires

from scipy.special import erf

import streamlit as st
import numpy as np
from matplotlib import pyplot as plt

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

# Configurations for execution.


def compute_ISI(spks):
    """
    Damien's code.
    """
    # hint spks is a 2D matrix, get a 1D Vector per neuron-id spike train.
    # [x for ind,x in enumerate(spks)]
    # spkList = [x for ind,x in enumerate(spks)]
    ISI = []
    for neurons in spks:
        ISI.append([j - i for i, j in zip(neurons[:-1], neurons[1:])])
    return np.asarray(ISI)
    # st.markdown(spkList)
    # st.pyplot()
    # pass
    # return an array of ISI_arrays.


def compute_ISI_CV(spks):
    ISIs = compute_ISI(spks)
    # hint
    # [x for ind,x in enumerate(spks)]
    pass
    # return a vector of scalars: ISI_CV


def average(ISI_CV):
    # use numpy to mean the vector of ISI_CVs
    # return a scalar.
    pass


def raster_plot(spks, stride=1, fig=None, color="b", alpha=1):
    """Generate raster plot of spiking activity.
    
    Parameters
    ----------
    
    spks : np.ndarray shape (num_neurons, timesteps)
        Spiking activity of neurons, a spike is indicated by a one    
    stride : int
        Stride for plotting neurons
    """
    num_time_steps = spks.shape[1]
    assert (
        stride < num_time_steps
    ), "Stride must be smaller than number of time steps"

    time_steps = np.arange(0, num_time_steps, 1)
    if fig is None:
        fig = plt.figure(figsize=(10, 5))
    timesteps = spks.shape[1]

    plt.xlim(-1, num_time_steps)
    plt.yticks([])

    plt.xlabel("Time steps")
    plt.ylabel("Neurons")

    for i in range(0, dim, stride):
        spike_times = time_steps[spks[i] == 1]
        plt.plot(
            spike_times,
            i * np.ones(spike_times.shape),
            linestyle=" ",
            marker="o",
            markersize=1.5,
            color=color,
            alpha=alpha,
        )

    return fig

def spikes_to_frame(dims, spks) -> (pd.DataFrame,dict):
    timesteps = num_time_steps = spks.shape[1]
    stride = 6
    time_steps = np.arange(0, num_time_steps, 1)

    assert (
        stride < num_time_steps
    ), "Stride must be smaller than number of time steps"

    spk_time_list = []
    spike_times_dic = {}
    for i in range(0, dim, stride):
        temp = [float(x) for x in time_steps[spks[i] == 1]]
        spike_times_dic[str(i)] = temp
    spk_time_list.append(spike_times_dic)
    spike_frame = pd.DataFrame(spk_time_list)

    return (spike_frame, spike_times_dic)

def plot3(spks_balanced) -> None:
    fig = raster_plot(spks=spks_balanced)
    st.pyplot(fig)


uploaded_file = st.file_uploader("Upload Spike Trains To Compute CV on.")

if uploaded_file is not None:
    spks_dict = pickle.loads(uploaded_file.read())
    st.write("spikes loaded")
    #st.write(spks_balanced)

    balanced_spikes = spks_dict["balanced"]
    critical_spikes = spks_dict["critical"]
    critical_fixed_spikes = spks_dict["critical_fixed"]

    for spkt in [balanced_spikes,critical_spikes,critical_fixed_spikes]:
        raster_plot(spkt)
    for spkt in [balanced_spikes,critical_spikes,critical_fixed_spikes]:
        result = compute_ISI_CV(spkt)
        st.markdown(result)

else:
    #from sim_param_imports import *

    st.markdown("No files where uploaded yet, so generating the data that make up those files... Please Download them when done with the Download link.")
    import sim_param_imports

    flatten_run_params = [(dim,num_steps) for dim in [50,100,200] for num_steps in 500,1000,2000]
    results_dic = {}
    my_bar = st.progress(0)

    for ind,(dim_global,num_steps) in enumerate(flatten_run_params):
        percent_complete = float(ind/len(flatten_run_params))
        
        my_bar.progress(percent_complete + 1)

        sim_param_imports.dim = dim_global


        (data_v_balanced, data_v_balanced, spks_balanced) = third_model_to_cache(
            sim_param_imports.network_params_balanced,num_steps
        )
        spks_critical,data_u_critical,data_v_critical,lif_network_critical,lif_params_critical = fourth_model(sim_param_imports.network_params_critical,num_steps)

        spks_critical_fixed = fifth_model_to_cache(num_steps,data_u_critical,data_v_critical,sim_param_imports.network_params_critical)


        spike_frame0, spike_times_balanced = spikes_to_frame(dim, spks_balanced)

        _ = plot3(spks_balanced)
        spike_frame1, spike_times_critical = spikes_to_frame(dim, spks_critical)
        _ = plot3(spks_critical)
        spike_frame2, spike_times_critical_fixed = spikes_to_frame(dim, spks_critical_fixed)
        _ = plot3(spks_critical_fixed)
        results_dic[dim_global,num_steps] = {"balanced":spike_times_balanced,"critical":spike_times_critical,"critical_fixed":spike_times_critical_fixed}
    
    st.download_button(
        "Download Spikes",

        data=pickle.dumps(results_dic),
        file_name="spks_balanced.pkl",
    )
