from common import ErrorType,MethodType
from problems import StokesProblemData
from solvers import StokesSolution
import numpy as np
from fenics import errornorm,Measure,assemble,inner,div
from typing import Dict

def u_l2_error(data: StokesProblemData, solution: StokesSolution, degree_rise: int = 3):
    return errornorm(data.u_exact,solution.u,norm_type="L2",degree_rise=degree_rise)

def u_h1_error(data: StokesProblemData, solution: StokesSolution, degree_rise: int = 3):
    return errornorm(data.u_exact,solution.u,norm_type="H10",degree_rise=degree_rise)

def p_l2_error(data: StokesProblemData, solution: StokesSolution, degree_rise: int=3):
    return errornorm(data.p_exact,solution.p,norm_type="L2",degree_rise=degree_rise)

def T_l2_error(data: StokesProblemData, solution: StokesSolution) -> float:
    error_tensor = data.T_exact - solution.T
    dx_mesh = Measure("dx",domain=data.mesh)
    error = assemble(inner(error_tensor,error_tensor)*dx_mesh)

    # Avoiding floating point sign errors
    return float(np.sqrt(max(0.0,error)))

def T_hdiv_error(data:StokesProblemData, solution: StokesSolution) -> float:
    error_div = -data.forcing_term - div(solution.T)

    dx_mesh = Measure("dx",domain=data.mesh)
    error = assemble(inner(error_div,error_div)*dx_mesh)
    error += (T_l2_error(data,solution)**2)

    # Avoiding floating point sign errors
    return float(np.sqrt(max(0.0,error)))


ERROR_DISPATCHER = {
    ErrorType.U_L2: lambda data, sol,**kw: u_l2_error(data,sol,degree_rise=kw.get("degree_rise",3)),
    ErrorType.U_H1: lambda data, sol,**kw: u_h1_error(data,sol,degree_rise=kw.get("degree_rise",3)),
    ErrorType.P_L2: lambda data, sol,**kw: p_l2_error(data,sol,degree_rise=kw.get("degree_rise",3)),
    ErrorType.T_L2: lambda data, sol,**kw: T_l2_error(data,sol),
    ErrorType.T_HDIV: lambda data, sol, **kw: T_hdiv_error(data,sol)
}

def compute_errors(data: StokesProblemData,solution: StokesSolution, method: MethodType, degree_rise: int = 3) -> Dict[ErrorType, float] :
    errs = {}
    for norm in method.accepted_norms:
        errs[norm] = ERROR_DISPATCHER[norm](data,solution,degree_rise=degree_rise)
    return errs