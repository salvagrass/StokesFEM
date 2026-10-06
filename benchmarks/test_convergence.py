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
    methods = [MethodType.TH,MethodType.MINI,MethodType.KS,MethodType.AFW]

    # Preparing the plotter
    plotter = ConvergencePlotter(methods,'h',"MMS",normalized=True)

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
        # Adding the error line to the plotter
        print(f"Adding the line for the plotter for {method}")
        plotter.plot_from_study(study)
    
    print(f"--- Adding reference lines to the plotter ---")
    h_vals = [1.0 / n for n in refs]
    for err_type in plotter.active_norms:
        for order in [1,2,3]:
            plotter.add_reference_slope(err_type,h_vals,order)

    print("--- Saving the figure in the output directory ---")
    save_path = f"{OUTPUT_DIR}/convergence.png"
    plotter.fig.savefig(save_path,dpi=300)


if __name__ == "__main__":
    test_convergence()