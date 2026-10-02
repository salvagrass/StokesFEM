import os
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from fenics import Constant, plot, File

from problems.obstacle import generate_problem
from solvers.factory import SolverFactory

def test_obstacle_all_solvers():
    print("1. Mesh Generation and Problem Data...")
    problem_data = generate_problem(n=30)
    
    # Methods to test
    methods_to_test = ["th", "mini", "ks","afw"]
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    for method in methods_to_test:
        print(f"\n--- {method.upper()} Test ---")
        
        print(f"Using SolverFactory...")
        solver = SolverFactory.create_solver(method, fluid_model=None, degree=2)
        
       
        print("Calling solve...")
        solution = solver.solve(problem_data,Constant(2.0))
        

        assert solution.u is not None, f" Velocity field not computed - {method}!"
        assert solution.p is not None, f"Pressure field not computed - {method}!"
        
    
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plot(solution.u, title=f"Velocity ({method.upper()})")
        
        plt.subplot(1, 2, 2)
        plot(solution.p, title=f"Pressure ({method.upper()})")
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/obstacle_solution_{method}.png", dpi=300)
        plt.close() # Chiude la figura per liberare memoria tra un'iterazione e l'altra
        print(f"Successfully saved result for '{method}' in '{output_dir}/'.")

if __name__ == "__main__":
    test_obstacle_all_solvers()