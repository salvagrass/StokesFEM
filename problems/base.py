from dataclasses import dataclass
from fenics import Mesh,MeshFunction
from typing import Dict,Any

@dataclass
class StokesProblemData:
    """Container for data of a Stokes Problem.
    
    Attributes:
        mesh (Mesh): The divided geometrical domain.
        boundary_markers (MeshFunction): Mesh function used to mark subsets of the boundaries.
        dirichlet_bcs (Dict[int, Any]): Dictionary mapping boundary IDs (markers) to 
            Dirichlet boundary conditions or expressions (e.g., prescribed velocity values).
        neumann_bcs (Dict[int, Any]): Dictionary mapping boundary IDs (markers) to 
            Neumann boundary conditions or surface traction expressions.
        forcing_term (Any): Volumetric source/body force term acting on the fluid (e.g., Constant or Expression).
    """
    mesh: Mesh
    boundary_markers: MeshFunction
    dirichlet_bcs: Dict[int,Any]
    neumann_bcs: Dict[int,Any]
    forcing_term: Any
