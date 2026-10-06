from solvers import SolverFactory,MethodType
from problems.cavity import generate_problem
from postprocessing import plots

from fenics import Constant

import os

OUTPUT_PATH = "results/cavity"

def test_cavity():

    print("--- Lid Driven Cavity Test ---")

    # Generating the problem
    data = generate_problem(n=20)
    factory = SolverFactory()

    # Setting up the output directory
    os.makedirs(OUTPUT_PATH,exist_ok=True)


    for method in [MethodType.HYBRID, MethodType.AFW,MethodType.RTCG]:
        print(f"--- Computing the solution for {method} ---")
        print("Creating the factory")
        solver = factory.create_solver(method=method,fluid_model=None,degree = 3)

        print("Solving the problem")
        solution = solver.solve(data=data,mu=Constant(1.0))
        assert solution.u is not None, f"Velocity field not computed - {method}!"
        assert solution.p is not None, f"Pressure field not computed - {method}!"

        fig, axes = plots.plot_combined(
            solution=solution,
            mode='streamlines',
            title_prefix=f"Cavity {method}",
            layout="horizontal",
            wireframe=True
        )

        print(f"Saving the results")

        save_path = f"{OUTPUT_PATH}/cavity_{method}.png"
        fig.savefig(save_path,dpi=300)

if __name__ == "__main__":
    test_cavity()

