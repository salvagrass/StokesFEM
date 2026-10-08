from problems import StokesProblemData
from common import MethodType
from solvers.base import StokesSolution,StokesSolver

from fenics import *
from ufl.coefficient import Coefficient


class AugmentedSolver(StokesSolver):

    def __init__(self, fluid_model, degree: int = 2):
        if degree < 1:
            raise ValueError("Degree too low for the method")
        super().__init__(fluid_model)
        self.degree = degree 

    
    def solve(self,data: StokesProblemData, mu: Coefficient) -> StokesSolution:

        #Spaces definition 
        cell = data.mesh.ufl_cell()
        S1 = FiniteElement("BDM",cell,degree=self.degree)
        S2 = FiniteElement("BDM",cell,degree=self.degree)

        #! Doesn't work with DG elements with degree - 1
        N1 = FiniteElement("BDM",cell,degree=self.degree)
        N2 = FiniteElement("BDM",cell,degree=self.degree)

        U = VectorElement("DG",cell, self.degree-1)

        # Constructing global space
        W = FunctionSpace(data.mesh,MixedElement([S1,S2,N1,N2,U]))

        # Setting up our elements
        t1,t2,r1,r2,u = TrialFunctions(W)
        s1,s2,e1,e2,v =TestFunctions(W)

        T = as_tensor([t1,t2])
        R = as_tensor([r1,r2])

        sigma = as_tensor([s1,s2])
        eta = as_tensor([e1,e2])

        # Setting up the Neumann Boundary
        bcs = []
        for bnd_id, bc_value in data.mixed_neumann_bcs.items():
            bcs.append(
                DirichletBC(W.sub(0), bc_value[0], data.boundary_markers, bnd_id)
            )
            bcs.append(
                DirichletBC(W.sub(1), bc_value[1], data.boundary_markers, bnd_id)
            )

        # Constructing the formulation
        lhs = (
            -inner(2*mu*sym(R),sym(eta)) * dx
            + inner(dev(T),dev(eta))* dx
            + inner(dev(sigma),dev(R)) * dx
            + dot(div(sigma),u) * dx
            + dot(div(T),v) * dx
        )

        rhs = -dot(data.forcing_term,v)*dx

        # Adding boundary terms
        ds = Measure(
            "ds", subdomain_data=data.boundary_markers, domain=data.mesh
        )
        n = FacetNormal(data.mesh)

        for bnd_id, bc_value in data.dirichlet_bcs.items():
            rhs += dot(dot(sigma, n), bc_value) * ds(bnd_id)

        # Solving the system and splitting the data:
        sol = Function(W)
        solve(lhs==rhs,sol,bcs)

        t1_sol,t2_sol,_,_,u_sol = sol.split(deepcopy=True)
        T_sol = as_tensor([t1_sol,t2_sol])

        p_expr = -0.5*tr(T_sol)
        p_space = FunctionSpace(data.mesh,"DG",self.degree-1)
        p_sol = project(p_expr,p_space)

        return StokesSolution(u_sol,p_sol,T_sol,sol,MethodType.AUG)
        
