from problems.obstacle import generate_problem
from solvers import SolverFactory,MethodType
from postprocessing import plots


from fenics import Constant

import os

OUTPUT_DIR = "results/obstacle"

def test_obstacle():
    # Generating the problem 
    problem_data = generate_problem(n=32)

    # Getting the list of available methods
    methods = list(MethodType)


    # Setting up the factory and the directory
    factory = SolverFactory()
    os.makedirs(OUTPUT_DIR,exist_ok=True)

    # Iterating on all of the methods
    for method in methods:
        solver = factory.create_solver(method=method,fluid_model=None)

        print(f"--- Solving with {method}---")
        solution = solver.solve(data=problem_data,mu=Constant(1.0))

        assert solution.u is not None, f"Velocity field not computed - {method}!"
        assert solution.p is not None, f"Pressure field not computed - {method}!"

        fig, axes = plots.plot_combined(
            solution=solution,
            mode='quiver',
            title_prefix= f"{method} - ",
            layout='vertical'
            )

        print(f"Saving the graph for {method} in {OUTPUT_DIR}")
        save_path = f"{OUTPUT_DIR}/obstacle_{method}.png"
        fig.savefig(save_path,dpi=300)

if __name__ == "__main__":
    test_obstacle()



        

        