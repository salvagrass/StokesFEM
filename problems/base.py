from dataclasses import dataclass
from typing import Dict, Optional
from fenics import Mesh,MeshFunction
from ufl.coefficient import Coefficient


@dataclass
class StokesProblemData:
    """Container for data and boundary value specifications of a Stokes problem.

    Attributes:
        mesh (Mesh): The discretized computational domain.
        boundary_markers (MeshFunction): Mesh function labeling boundary facets
            with integer subdomain markers.
        dirichlet_bcs (Dict[int, Coefficient]): Mapping of boundary subdomain IDs
            to prescribed Dirichlet values (e.g., velocity profiles as Constant,
            Expression, or Function).
        neumann_bcs (Dict[int, Coefficient]): Mapping of boundary subdomain IDs
            to natural boundary tractions (e.g., normal stress or surface force).
        mixed_neumann_bcs (Dict[int, Coefficient]): Mapping of boundary subdomain IDs
            to traction or pseudo-stress terms specific to mixed or augmented formulations.
        forcing_term (Coefficient): Volumetric source/body force term acting on the
            fluid (e.g., gravity or manufactured source term).
        u_exact (Optional[Coefficient]): Analytical or high-fidelity reference velocity field
            used for error evaluation and convergence verification (MMS). Defaults to None.
        p_exact (Optional[Coefficient]): Analytical or high-fidelity reference pressure field
            used for error evaluation and convergence verification (MMS). Defaults to None.
    """

    mesh: Mesh
    boundary_markers: MeshFunction
    dirichlet_bcs: Dict[int, Coefficient]
    neumann_bcs: Dict[int, Coefficient]
    mixed_neumann_bcs: Dict[int, Coefficient]
    forcing_term: Coefficient

    u_exact: Optional[Coefficient] = None
    p_exact: Optional[Coefficient] = None
    T_exact: Optional[Coefficient] = None