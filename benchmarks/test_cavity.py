from solvers import SolverFactory,MethodType
from solvers.strategies import *
from commons import PowerLawModel,LinearModel
from problems.cavity import generate_problem
from postprocessing import plots


import os

OUTPUT_PATH = "results/cavity"

def test_cavity():

    print("--- Lid Driven Cavity Test ---")

    # Generating the problem
    data = generate_problem(n=32)
    model = PowerLawModel(K=0.5,r=1.5)

    strategy = HybridStrategy(num_iter_p=20)
    # Setting up the output directory
    os.makedirs(OUTPUT_PATH,exist_ok=True)


    for method in list(MethodType):
        print(f"--- Computing the solution for {method} ---")
        print("Creating the factory")
        solver = SolverFactory.create_solver(method,model)

        print("Solving the problem")
        solution = solver.solve(data=data,strategy=strategy)
        assert solution.u is not None, f"Velocity field not computed - {method}!"
        assert solution.p is not None, f"Pressure field not computed - {method}!"

        fig, axes = plots.plot_combined(
            solution=solution,
            mode='streamlines',
            title_prefix=f"Cavity {method} - Power Law K=0.5, r = 1.5",
            layout="horizontal",
        )

        print(f"Saving the results")

        save_path = f"{OUTPUT_PATH}/cavity_{method}.png"
        fig.savefig(save_path,dpi=300)

if __name__ == "__main__":
    test_cavity()

