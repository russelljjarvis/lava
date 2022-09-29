import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

files = glob.glob("*.p")
st.markdown(files)
list_of_spike_file_contents = []
for f in files:
    with open(f,"rb") as f:
        list_of_spike_file_contents.append(pickle.load(f))



def wrangle_frame(frame)->None:
    for c in frame.columns:
        frame[c].values[:] = pd.Series(frame[c])
    st.write(frame)


def plot_raster(spike_dict)->None:
    fig = plt.figure()
    list_of_lists = []
    for ind,(neuron_id,times) in enumerate(spike_dict.items()):
        list_of_lists.append(times)
    plt.eventplot(list_of_lists)
    st.pyplot(fig)

def wrangle(spike_dict)->[[]]:
    list_of_lists = []
    maxt=0
    for ind,(neuron_id,times) in enumerate(spike_dict.items()):
        list_of_lists.append(times)
        if np.max(times)> maxt:
            maxt = np.max(times)
    st.markdown("Dimensions are: ")
    st.markdown(np.shape(list_of_lists))
    st.markdown(maxt)
    return list_of_lists



st.markdown("[Link to Code That Generated The Plots:](https://github.com/russelljjarvis/lava/blob/main/tutorials/end_to_end/tutorial02_excitatory_inhibitory_network.ipynb)")

spikes_in_list_of_lists_of_lists = []

for i in list_of_spike_file_contents:
    st.markdown("# The data as a table:")
    wrangle_frame(i[0])

    st.markdown("# The raster plot:")
    plot_raster(i[1])
    spikes_in_list_of_lists_of_lists.append(wrangle(i[1]))


def compute_ISI(spks):
    """
    Damien's code.
    """
    # hint spks is a 2D matrix, get a 1D Vector per neuron-id spike train.
    # [x for ind,x in enumerate(spks)]
    # spkList = [x for ind,x in enumerate(spks)]

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

