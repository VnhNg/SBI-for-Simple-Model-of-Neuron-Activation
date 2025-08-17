import bayesflow as bf
import keras
import numpy as np

# %%
def LIF_model(tau_m, R, amp, noise_std, V_rest=-65.0, V_th=-50.0, V_reset=-65.0):
    """Simulate Voltage trajectory
    tau_m: membrane time constant
    R: membrane resistance
    amp: amplitude of rectangular input current in high status
    noise_std: noisy"""
    n_steps = 1000
    dt = 1.0  # ms

    V = V_rest * np.ones(n_steps)
    I = np.zeros(n_steps)
    spikes = np.zeros(n_steps)

    # Rectangular pulse input + noise
    I[300:700] = amp
    I += noise_std * np.random.randn(n_steps)

    for t in range(1, n_steps):
        dV = dt * (-(V[t - 1] - V_rest) + R * I[t]) / tau_m
        V[t] = V[t - 1] + dV
        if V[t] >= V_th:
            V[t] = V_reset
            spikes[t] = 1  # mark spike

    return I, V, spikes

def prior():
    """Simulate tau_m and R indirectly"""
    # Specific capacitance: ~10 nF/mm², tightly distributed
    c_m = np.random.lognormal(mean=np.log(10), sigma=0.05)  # nF/mm²

    # Surface area: 0.01–0.1 mm²
    A = 0.05  # np.random.uniform(0.03, 0.07)  # mm²

    # Specific resistance: log-normal around 1 MΩ·mm²
    r_m = np.random.lognormal(mean=np.log(1.0), sigma=0.5)  # MΩ·mm²

    # Derived values
    tau_m = r_m * c_m  # ms (MΩ·nF = ms)
    R = r_m / A  # MΩ

    return dict(tau_m=tau_m, R=R)


def likelihood(tau_m, R):
    "Vary input current and simulate Voltage trajectory"
    tau_m = tau_m
    R = R

    # Input current with noisy
    amp = 1  # np.random.uniform(0.05, 1)
    noise = 0.05  # np.random.uniform(0.005, 0.1)

    # Generate data
    I, V, spikes = LIF_model(tau_m, R, amp, noise)
    return dict(I=I, V=V, spikes=spikes)


def simulator():
    return bf.make_simulator([prior, likelihood])
# %%
import matplotlib.pyplot as plt
# Parameters for small tau_m
small_tau_values = [10.0]  # ms
R = 1
amp = 2.0
noise_std = 1.5

plt.figure(figsize=(8, 4))

for tau in small_tau_values:
    I, V, spikes = LIF_model(tau, R, amp, noise_std)
    plt.plot(V, label=f"tau_m = {tau} ms")

plt.axvline(300, color='k', linestyle='--', alpha=0.5, label='Step on')
plt.axvline(700, color='k', linestyle='--', alpha=0.5, label='Step off')
plt.xlabel("Time step (1 ms each)")
plt.ylabel("Membrane potential (mV)")
plt.title("Voltage traces for small tau_m with dt = 1 ms")
plt.legend()
plt.tight_layout()
plt.show()
# %%

def show_sample_observation(sample, index, V_reset=-65.0):
    I = sample["I"][index]
    V = sample["V"][index]
    spikes = sample["spikes"][index]
    time = np.linspace(0, 1.0, len(I))

    plt.figure(figsize=(12, 8))

    # Input current
    plt.subplot(2, 1, 1)
    plt.plot(time, I, color='tab:blue')
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("Current (nA)", fontsize=20)
    plt.title("Input Current", fontsize=30)
    plt.tick_params(axis='both', labelsize=16)  # numbers on axis

    # Membrane potential
    plt.subplot(2, 1, 2)
    plt.plot(time, V, label="Membrane potential", color='tab:blue')
    plt.axhline(-50, color='red', linestyle='--', linewidth=1, label='Threshold')

    # Spike lines
    spike_times = time[np.where(spikes)[0]]
    for t in spike_times:
        plt.vlines(t, ymin=V_reset, ymax=-45.0, color='tab:blue', linewidth=1)

    plt.ylim(-70, -40)
    plt.xlabel("Time (s)", fontsize=20)
    plt.ylabel("Voltage (mV)", fontsize=20)
    plt.title("Membrane Potential with Spike Lines", fontsize=30)
    plt.tick_params(axis='both', labelsize=16)  # numbers on axis

    plt.tight_layout()
    plt.show()
# %%

class SummaryNetwork(bf.networks.SummaryNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.network = keras.Sequential(
            [
                keras.layers.Dense(400, activation="relu"),
                keras.layers.Dense(200, activation="relu"),
                keras.layers.Dense(100, activation="relu"),
                keras.layers.Dense(50, activation="relu"),
            ]
        )

    def call(self, x, **kwargs):
        summary = self.network(x, training=kwargs.get("stage") == "training")

        return summary


def adapter():
    adapter = (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(["tau_m", "R"], into="inference_variables")
        .concatenate(["V"], into="summary_variables")
    )

    return adapter


def inference_network():
    return bf.networks.CouplingFlow()


def lif_workflow():
    workflow = bf.BasicWorkflow(
        inference_network=inference_network(),
        summary_network=SummaryNetwork(),
        adapter=adapter(),
        simulator=simulator(),
    )

    return workflow
# %%

training_sample = simulator().sample(1000)
validation_sample = simulator().sample(1000)
# %%

show_sample_observation(training_sample, index=8)
show_sample_observation(validation_sample, index=12)
# %%

wf = lif_workflow()
history = wf.fit_offline(
    data=training_sample, 
    epochs=500, 
    batch_size=64, 
    validation_data=validation_sample
    )
# %%

bf.diagnostics.plots.loss(
    history,
    figsize=(15, 6),        
    legend_fontsize=20,
    label_fontsize=30,
    title_fontsize=0,      
    lw_train=2.5,
    lw_val=2.5
    )
# %%

"Simulation for testing"
# Set the number of posterior draws you want to get
num_samples = 100

# Simulate validation data (unseen during training)
val_sims = simulator().sample(2000)

# Obtain num_samples samples of the parameter posterior for every validation dataset
post_draws = wf.sample(conditions=val_sims, num_samples=num_samples)
post_draws.keys()
print(post_draws["tau_m"].shape)
# %%
"Diagnostics"
bf.diagnostics.plots.calibration_histogram(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=['tau_m', 'R']
)

bf.diagnostics.plots.pairs_posterior(
    estimates=post_draws, 
    targets=val_sims,
    dataset_id=0,
    variable_names=['tau_m', 'R'],
)

bf.diagnostics.plots.recovery(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=['tau_m', 'R']
)

bf.diagnostics.plots.z_score_contraction(
    estimates=post_draws, 
    targets=val_sims,
    variable_names=['tau_m', 'R']
)

bf.diagnostics.plots.calibration_ecdf(
    estimates=post_draws, 
    targets=val_sims,
    difference=True,
    variable_names=['tau_m', 'R']
)
# %%
# GRU
class GRU(bf.networks.SummaryNetwork):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.gru = keras.layers.GRU(64, dropout=0.1)
        self.summary_stats = keras.layers.Dense(8)
        
    def call(self, time_series, **kwargs):
        """Compresses time_series of shape (batch_size, T, 1) into summaries of shape (batch_size, 8)."""

        summary = self.gru(time_series, training=kwargs.get("stage") == "training")
        summary = self.summary_stats(summary)
        return summary


def adapterGRU():
    return (
        bf.adapters.Adapter()          # or bf.Adapter() depending on your BF version
        .convert_dtype("float64", "float32")
        .as_time_series(["V"])         # <-- add this: makes V shape (B, T, 1)
        .concatenate(["V"], into="summary_variables")
        .concatenate(["tau_m", "R"], into="inference_variables")
    )

# %%
# LSTM
def adapterLSTM():
    return (
            bf.adapters.Adapter()
            .convert_dtype("float64", "float32")
            .standardize(include=["tau_m", "R"])
            .concatenate(["tau_m", "R"], into="inference_variables")
            .as_time_series(["I", "V", "spikes"])
            .concatenate(["V", "I", "spikes"], into="summary_variables")
        )





