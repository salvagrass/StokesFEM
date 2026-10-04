from problems.base import StokesProblemData
from fenics import UnitSquareMesh,MeshFunction,CompiledSubDomain,Constant

def generate_problem(n: int = 20) -> StokesProblemData:
    """Generate the classic Lid-Driven Cavity benchmark problem for Stokes flow.

    The computational domain is the unit square [0, 1] x [0, 1]. A constant
    horizontal tangential velocity is prescribed on the top lid (y = 1),
    while no-slip boundary conditions are enforced on the remaining three walls.
    Body forces are set to zero.

    Args:
        n (int, optional): Number of subdivisions along each spatial direction.
            Defaults to 20.

    Returns:
        StokesProblemData: Configured problem container with discretized domain,
            facet boundary markers, Dirichlet boundary conditions, and zero forcing.
    """
    mesh = UnitSquareMesh(n,n,'crossed')
    boundary_markers = MeshFunction(
        'size_t',
        mesh,mesh.topology().dim() -1,
        0
    )

    CompiledSubDomain("on_boundary and near(x[1],1)").mark(boundary_markers,1)
    CompiledSubDomain("on_boundary and not near(x[1],1)").mark(boundary_markers,2)

    upper = Constant((1.0,0.0))
    not_upper = Constant((0.0,0.0))
    
    dirichlet_bcs = {
        1: upper,
        2: not_upper
    }

    forcing_term = Constant((0.0,0.0))

    return StokesProblemData(mesh,boundary_markers,dirichlet_bcs,{},{},forcing_term)    
