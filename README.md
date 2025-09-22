# LIF + BayesFlow Posterior Estimation

Neurons are the fundamental signaling units of the nervous system. They communicate via electrical impulses, the so-called action potentials, which are triggered by the opening and closing of voltage-gated ion channels located across the neuron’s membrane.
The fundamental understanding of how action potentials are generated as a function of the input current, ion conductances, as well as membrane potentials and capacitances, was laid in the 1930s and 1940s by scientist such as Alan Hodgkin and Andrew Huxley, who performed extensive
experiments on giant squid axons.
In this project, the task is to implement a simple model of neuron activation - the leaky integrate-and-fire (LIF) model (see [1] and additional self-researched references), which is less biophysically detailed than the full Hodgkin-Huxley model, but still captures the essential elements
neuron behaviour like membrane potential integration, leakage, and spiking.

This repository implements a **Leaky Integrate-and-Fire (LIF)** neuron model and applies **BayesFlow** for simulation-based inference of the membrane time constant (τm) and resistance (R).



