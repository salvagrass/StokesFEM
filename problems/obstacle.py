from fenics import *
from mshr import *
from problems.base import StokesProblemData

def generate_problem(n: int=20) -> StokesProblemData:
    """Generates the computational domain, mesh, and boundary conditions for a 2D Stokes flow.

    The problem models laminar flow through a channel containing a circular obstacle:
      - Domain: [-1.0, 4.0] x [-1.0, 1.0] with a centered circular obstacle removed at (0, 0).
      - Inflow (x = -1.0): Parabolic horizontal velocity profile.
      - Walls & Obstacle: No-slip (zero velocity) condition.
      - Outflow (x = 4.0): Natural boundary condition (do-nothing / zero traction).
      - Body force: Zero.

    Args:
        n (int, optional): Mesh resolution parameter passed to mshr's `generate_mesh`.
            Higher values yield a finer triangulation. Defaults to 20.

    Returns:
        StokesProblemData: A container holding the mesh, boundary facet markers,
            Dirichlet boundary conditions mapping, Neumann conditions mapping,
            and the zero body force vector.
    """
    
    # Generating the mesh for the problem
    obstacle = Circle(Point(0.0,0.0),0.2)
    domain = Rectangle(Point(-1.0,-1.0),Point(4.0,1.0))

    geometry = domain - obstacle
    
    mesh = generate_mesh(geometry, n)

    # Applying the boundary markers for this specific problem 
    boundary_markers = MeshFunction('size_t',mesh,mesh.topology().dim()-1,0)

    CompiledSubDomain("on_boundary and near(x[0],-1)").mark(boundary_markers,1) # Inflow Boundary
    CompiledSubDomain("on_boundary and not near(x[0], -1.0) and not near(x[0], 4.0)").mark(boundary_markers,2) # Rigid Boundary
    CompiledSubDomain("on_boundary and near(x[0], 4.0)").mark(boundary_markers,3) # Outflow Boundary

    u_inflow = Expression(('(1 - x[1] * x[1])', '0'), degree=2)
    u_rigid = Constant((0.0, 0.0))

    dirichlet_bcs = {
        1: u_inflow,
        2: u_rigid
    }

    traction_vec = Constant((0.0,0.0))
    neumann_bcs = {
        3: traction_vec
    }

    mixed_neumann_bcs = {3: [Constant((0.0,0.0)),Constant((0.0,0.0))]}

    return StokesProblemData(mesh,
                             boundary_markers,
                             dirichlet_bcs,
                             neumann_bcs,
                             mixed_neumann_bcs,
                             Constant((0.0, 0.0))
                             )
