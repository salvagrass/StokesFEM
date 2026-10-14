from problems.base import StokesProblemData
from commons import FluidModel,LinearModel
from fenics import (Mesh,
                    UnitSquareMesh,
                    Expression,
                    CompiledSubDomain,
                    MeshFunction,
                    Identity,
                    VectorFunctionSpace,
                    SpatialCoordinate,
                    sin,
                    cos,
                    as_vector,
                    sym,
                    grad,
                    div,
                    project,
                    pi
                    )


def generate_problem(n: int = 20, model: FluidModel = LinearModel(1.0)) -> StokesProblemData:
    """Generate an analytical Method of Manufactured Solutions (MMS) Stokes problem.

    The computational domain is the unit square [0, 1] x [0, 1]. A smooth
    manufactured solution with known velocity and pressure fields is imposed:
    Dirichlet boundary conditions are enforced on the bottom and left edges,
    while Neumann tractions are prescribed on the top and right edges.

    Args:
        n (int, optional): Number of mesh subdivisions per spatial direction.
            Defaults to 20.

    Returns:
        StokesProblemData: Configured problem container with domain mesh,
            boundary facets, boundary condition mappings, forcing field,
            and exact analytical solutions.
    """
    mesh = UnitSquareMesh(n,n,'crossed')

    boundary_markers = MeshFunction(
        'size_t',
        mesh,mesh.geometric_dimension()-1,
        0
        )

    # Subdomain classification:
    # 1 -> Top boundary (y = 1)
    # 2 -> Right boundary (x = 1)
    # 3 -> Left and bottom boundaries (x = 0 or y = 0)
    CompiledSubDomain("on_boundary and near(x[1],1)").mark(boundary_markers,1)
    CompiledSubDomain("on_boundary and near(x[0],1)").mark(boundary_markers,2)
    CompiledSubDomain("on_boundary and (near(x[0],0) or near(x[1],0))").mark(boundary_markers,3)

    # Load analytical solution fields, body force, and standard surface tractions and compute boundary conditions
    (u_exact, 
     p_exact,
     T_exact, 
     forcing_term,
     dirichlet_bcs,
     neumann_bcs,
     mixed_neumann_bcs ) = _compute_data(mesh, model)
   
    return StokesProblemData(
        mesh,
        boundary_markers,
        dirichlet_bcs,
        neumann_bcs,
        mixed_neumann_bcs,
        forcing_term,
        u_exact,
        p_exact,
        T_exact
    )
    
def _compute_data(mesh: Mesh, model: FluidModel):
    u_exact = Expression((
            '(1.0-cos(pi*x[0])) * sin(pi*x[1])',
            'sin(pi*x[0]) * (cos(pi*x[1])-1.0)'
                        ), degree=6)
    p_exact = Expression(
            '-0.25 * (cos(2*pi*x[0]) + cos(2*pi*x[1]))',
            degree=6)

    # Setting up automatic computation of the problem
    x,y = SpatialCoordinate(mesh)
    u_ufl = as_vector((
        (1.0 - cos(pi * x)) * sin(pi * y),
        sin(pi * x) * (cos(pi * y) - 1.0),
    ))
    p_ufl = -0.25 * (cos(2 * pi * x) + cos(2 * pi * y)) 


    
    dirichlet_bcs = {
        3: u_exact
    }


    # Automatic differentiation
    I = Identity(mesh.geometric_dimension())
    mu_exact = model.mu(u_ufl)

    T_exact = -p_ufl*I + 2*mu_exact*sym(grad(u_ufl))
    forcing_term = -div(T_exact)

    # Computing the Neumann Boundary conditions automatically
    gNt = as_vector((T_exact[0,1],T_exact[1,1])) # N = (0,1), so Tn = (T01,T11)
    gNr = as_vector((T_exact[0,0],T_exact[1,0])) # N = (1,0), so Tn = (T00,T10)

    neumann_bcs = {
        1: gNt,
        2: gNr
    }


    # the DirichletBC function only accepts C++ expressions
    
    first_row = as_vector((T_exact[0,0],T_exact[0,1]))
    second_row = as_vector((T_exact[1,0],T_exact[1,1]))
    
    proj_space = VectorFunctionSpace(mesh,"CG",4) # We project on a space of degree 4 to have negligible error in the solvers
    proj_1 = project(first_row,proj_space)
    proj_2 = project(second_row,proj_space)

    T_boundary = [proj_1, proj_2]

    mixed_neumann_bcs = {
        1: T_boundary,
        2: T_boundary
    }
    
    return (u_exact,
            p_exact,
            T_exact,
            forcing_term,
            dirichlet_bcs,
            neumann_bcs,
            mixed_neumann_bcs)