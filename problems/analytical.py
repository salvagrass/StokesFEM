from problems.base import StokesProblemData
from fenics import Expression,UnitSquareMesh,CompiledSubDomain,MeshFunction


def generate_problem(n: int = 20) -> StokesProblemData:
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

    # Load analytical solution fields, body force, and standard surface tractions
    u_exact, p_exact, forcing_term, gNt,gNr = _compute_data()

    dirichlet_bcs = {
        3: u_exact
    }

    neumann_bcs = {
        1: gNt,
        2: gNr
    }

    # Decomposed traction components for mixed/dual Stokes formulations (e.g. stress-velocity)
    g1_top = Expression((
        "0.0", 
        "-pi * (cos(pi*x[0]) + 1.0)"
    ), degree=2)
    g2_top = Expression((
        "0.0", 
        "0.25 * (cos(2*pi*x[0]) + cos(2*pi*x[1]))"
    ), degree=2)

    g1_right = Expression((
        "0.25 * (cos(2*pi*x[0]) + cos(2*pi*x[1]))", 
        "0.0"
    ), degree=2)
    g2_right = Expression((
        "pi * (cos(pi*x[1]) + 1.0)", 
        "0.0"
    ), degree=2)

    mixed_neumann_bcs = {
        1: [g1_top, g2_top],
        2: [g1_right, g2_right]
    }

    return StokesProblemData(
        mesh,
        boundary_markers,
        dirichlet_bcs,
        neumann_bcs,
        mixed_neumann_bcs,
        forcing_term,
        u_exact,
        p_exact
    )
    
def _compute_data():
    u_exact = Expression((
            '(1.0-cos(pi*x[0])) * sin(pi*x[1])',
            'sin(pi*x[0]) * (cos(pi*x[1])-1.0)'
                        ), degree=6)
    p_exact = Expression(
            '-0.25 * (cos(2*pi*x[0]) + cos(2*pi*x[1]))',
            degree=6)
    f = Expression((
            '-2*pi*pi*cos(pi*x[0])*sin(pi*x[1]) + 0.5*pi*sin(2*pi*x[0]) + pi*pi*sin(pi*x[1])',
            '2*pi*pi*sin(pi*x[0])*cos(pi*x[1]) + 0.5*pi*sin(2*pi*x[1]) - pi*pi*sin(pi*x[0])'
                   ), degree=2)
    
    gNt = Expression((
            '-pi * (cos(pi*x[0]) + 1.0)', 
            '0.25 * (cos(2*pi*x[0]) + cos(2*pi*x[1]))'
        ), degree=2)
    
    gNr = Expression((
            '0.25 * (cos(2*pi*x[0]) + cos(2*pi*x[1]))', 
            'pi * (cos(pi*x[1]) + 1.0)'  
        ), degree=2)
    return u_exact,p_exact,f,gNt,gNr