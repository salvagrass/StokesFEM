from common import MethodType
from problems.analytical import generate_problem
from solvers import SolverFactory
from postprocessing import ConvergenceStudy,ConvergencePlotter,compute_errors

from fenics import Constant

import os
OUTPUT_DIR = "results/analytical"

def test_convergence():

    # Setting up the increments
    refs = [2,4,8,16,32]

    # Preparing the output directory
    os.makedirs(OUTPUT_DIR,exist_ok=True)

    # Preparing the list of methods
    methods = [MethodType.TH,MethodType.HYBRID,MethodType.AFW,MethodType.AUG]

    # Preparing the plotter
    plotter = ConvergencePlotter('h',"MMS",normalized=True)
    studies = []
    for method in methods:
        print(f"--- Setting up the problem for {method} ---")
        study = ConvergenceStudy(method)
        for n in refs:
            print(f"Setting up the problem with n = {n}")
            # Generating the problem, and the solver
            data = generate_problem(n)
            solver = SolverFactory.create_solver(method,None,degree=2)

            # Solving the problem and computing errors
            print(f"Attempting to solve the problem and compute errors")
            solution = solver.solve(data,mu=Constant(1.0))  
            assert solution.u is not None, f"Velocity field not computed - {method}!"
            assert solution.p is not None, f"Pressure field not computed - {method}!"
            errors = compute_errors(data,solution,method)
            study.add_error_data(errors,1/n)
        # Printing the convergence EOC Table for the method
        study.print_table()
        print(f"Adding the study to the collection")
        studies.append(study)

    figures = plotter.plot_from_studies(studies,reference_orders=[1,2,3])
    print(f"--- Adding reference lines to the plotter ---")
    for error_type in figures:
        fig = figures[error_type]
        print(f"--- Saving figure for {error_type} in the output directory ---")
        save_path = f"{OUTPUT_DIR}/convergence - {error_type}.png"
        fig.savefig(save_path,dpi=300)
if __name__ == "__main__":
    test_convergence()