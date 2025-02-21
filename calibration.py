import wntr
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.default import DefaultMultiObjectiveTermination
import matplotlib.pyplot as plt

# Load the hydraulic model
def load_model(filename):
    return wntr.network.WaterNetworkModel(filename)

# Load observed pressures from CSV file
def load_observed_pressures(filename):
    df = pd.read_csv(filename, index_col=0, delimiter=',')
    df.index = pd.to_datetime(df.index, format='%H:%M').hour + pd.to_datetime(df.index, format='%H:%M').minute / 60  # Convert time to fractional hours
    return df

# Define optimization problem
class HydraulicCalibration(Problem):
    def __init__(self, model, observed_pressures):
        self.model = model
        self.observed_pressures = observed_pressures
        self.pipes = list(model.pipe_name_list)
        super().__init__(n_var=len(self.pipes), n_obj=1, n_constr=0,
                         xl=0.1, xu=5, type_var=float)
    
    def _evaluate(self, X, out, *args, **kwargs):
        errors = []
        modelled_pressures_list = []
        for i in range(X.shape[0]):
            # Set roughness coefficients
            for j, pipe in enumerate(self.pipes):
                self.model.get_link(pipe).roughness = X[i, j]
            
            # Run simulation
            sim = wntr.sim.EpanetSimulator(self.model)
            results = sim.run_sim()
            
            # Compute error
            modelled_pressures = results.node['pressure'][self.observed_pressures.columns]
            error = np.abs(modelled_pressures.values - self.observed_pressures.values).sum()
 #           print(error)
            errors.append(error)
            modelled_pressures_list.append(modelled_pressures)
        out["F"] = np.array(errors).reshape(-1, 1)
        self.modelled_pressures_list = modelled_pressures_list

# Load model and observed pressures
model = load_model("network_original_G1.inp")
observed_pressures = load_observed_pressures("observed_pressures.csv")

# Define problem
problem = HydraulicCalibration(model, observed_pressures)

# Configure and run NSGA2 optimization
algorithm = NSGA2(
    pop_size=100,  # Reduced population for faster execution
    sampling=LHS(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

termination = DefaultMultiObjectiveTermination()
result = minimize(problem, algorithm, termination, verbose=True)

# Extract best solution
best_index = np.argmin(result.F)
best_solution = result.X

# Display optimized roughness coefficients
optimized_roughness = pd.DataFrame({'Pipe': problem.pipes, 'Optimized Roughness': best_solution})
print(optimized_roughness)

optimized_roughness.plot()
# Plot observed vs. modeled pressures for best solution
best_modelled_pressures = problem.modelled_pressures_list[best_index]

for node in observed_pressures.columns:
    plt.figure()
    plt.plot(observed_pressures.index, observed_pressures[node], label="Observed", marker='o', linestyle='None')
    plt.plot(best_modelled_pressures.index/3600, best_modelled_pressures[node], label="Modeled", linestyle='--')
    plt.xlabel("Time (hours)")
    plt.ylabel("Pressure (m)")
    plt.xticks(np.arange(0, 24, 2))  # Set x-axis ticks every 2 hours
    plt.title(f"Observed vs. Modeled Pressure at Node {node}")
    plt.legend()
    plt.grid()
    plt.show()


# Load the water network model
model = wntr.network.WaterNetworkModel("network_original_G1.inp")

# Get pipe diameters as a dictionary {pipe_name: diameter}
pipe_diameters = model.query_link_attribute("diameter").to_dict()

# Plot the network using WNTR's built-in function
wntr.graphics.plot_network(
    model,
    link_attribute=pipe_diameters,  # Pipe diameters as link attribute
    node_attribute='elevation',  # Pipe diameters as link attribute
    link_cmap="viridis",  # Use the Viridis colormap
    link_colorbar_label="Diameter (m)",  # Label for the colorbar
    link_width=2,  # Adjust pipe width for better visibility
    node_labels=True,
    link_labels=False,
    title="Water Distribution Network - Pipe Diameters"
)



import matplotlib.pyplot as plt

# Set 'Pipe' column as index
optimized_roughness = optimized_roughness.set_index('Pipe')

# Plotting
ax = optimized_roughness.plot(kind='bar', title='Final roughness coefficients', rot=0)

# Customizations
plt.xlabel("Pipe")
plt.ylabel("Optimized Roughness")
plt.tight_layout()

# Display values on top of bars
for p in ax.patches:
    ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.005)) #Round to 2 decimals

plt.show()