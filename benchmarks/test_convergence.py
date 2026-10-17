from commons import MethodType,CarreauModel,PowerLawModel,LinearModel,ErrorType
from problems.analytical import generate_problem
from solvers import SolverFactory
from solvers.strategies import *
from postprocessing import ConvergenceStudy,ConvergencePlotter,compute_errors


import os
OUTPUT_DIR = "assets/"

def test_convergence():

    # Constructing linear model
    model = PowerLawModel(K=0.5,r=2.5)
    strategy = HybridStrategy()
    # Setting up the increments
    refs = [2,4,8,16,32,64]

    # Preparing the output directory
    os.makedirs(OUTPUT_DIR,exist_ok=True)

    # Preparing the list of methods
    methods = [MethodType.TH,MethodType.AUG_BDM,MethodType.AFW]

    # Preparing the plotter
    plotter = ConvergencePlotter('h',prefix="MMS - Power Law K=0.5, r=2.5",normalized=True)
    studies = []
    for method in methods:
        print(f"--- Setting up the problem for {method} ---")
        study = ConvergenceStudy(method)
        for n in refs:
            print(f"Setting up the problem with n = {n}")
            # Generating the problem, and the solver
            data = generate_problem(n,model)
            solver = SolverFactory.create_solver(method,model)

            # Solving the problem and computing errors
            print(f"Attempting to solve the problem and compute errors")
            solution = solver.solve(data,strategy)  
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
    for error_type in [ErrorType.U_L2,ErrorType.T_L2]:
        fig = figures[error_type]
        print(f"--- Saving figure for {error_type} in the output directory ---")
        save_path = f"{OUTPUT_DIR}/convergence - {error_type}.png"
        fig.savefig(save_path,dpi=300,bbox_inches="tight",pad_inches=0.1)
if __name__ == "__main__":
    test_convergence()