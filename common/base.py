from typing import Tuple,List
from enum import Enum

class ErrorType(str, Enum):
    """Supported Sobolev and Lebesgue discretization error norm metrics.

    Inherits from (str, Enum) for direct dictionary key compatibility,
    """

    U_L2 = "u_l2"
    U_H1 = "u_h1"
    P_L2 = "p_l2"
    T_L2 = "T_l2"
    T_HDIV = "T_hdiv"

    def __str__(self) -> str:
        return self.value

    @property
    def label(self) -> Tuple[str, str]:
        """Provides mathematical labels for plotting and CLI output.

        Returns:
            Tuple[str, str]: (LaTeX math symbol, plain text CLI label).
        """
        labels = {
            ErrorType.U_L2: (r"\|u - u_h\|_{L^2}", "L2(u)"),
            ErrorType.U_H1: (r"\|u - u_h\|_{H^1}", "H1(u)"),
            ErrorType.P_L2: (r"\|p - p_h\|_{L^2}", "L2(p)"),
            ErrorType.T_L2: (r"\|\mathbf{T} - \mathbf{T}_h\|_{L^2}", "L2(T)"),
            ErrorType.T_HDIV: (
                r"\|\mathbf{T} - \mathbf{T}_h\|_{H(\mathrm{div})}",
                "Hdiv(T)",
            ),
        }

        return labels[self]


class MethodType(str, Enum):
    """Enumeration of supported mixed finite element spatial discretizations.

    Inherits from (str, Enum) to ensure seamless serialization, terminal output,
    and dictionary key usage across metrics and plotting modules.
    """

    TH = "Taylor-Hood"
    MINI = "Mini"
    KS = "Kouhia-Stenberg"
    AFW = "Arnold-Falk-Winther"
    RTCG = "Raviart-Thomas"
    HYBRID = "Hybrid"
    AUG = "Augmented"

    def __str__(self) -> str:
        return self.value

    @property
    def style(self) -> Tuple[str, str]:
        """Matplotlib styling: (color, marker)."""
        styles = {
            MethodType.TH: ("tab:blue", "o"),
            MethodType.MINI: ("tab:orange", "s"),
            MethodType.KS: ("tab:green", "^"),
            MethodType.AFW: ("tab:red", "D"),
            MethodType.RTCG: ("tab:purple", "v"),
            MethodType.HYBRID: ("tab:brown", "p"),
            MethodType.AUG: ("tab:olive", "X"),
        }
        return styles[self]

    @property
    def accepted_norms(self) -> List[ErrorType]:
        """Canonical norms expected by mathematical a priori error bounds."""
        primal_norms = [
            ErrorType.U_L2,
            ErrorType.U_H1,
            ErrorType.P_L2,
            ErrorType.T_L2,
        ]
        mixed_norms = [
            ErrorType.U_L2,
            ErrorType.P_L2,
            ErrorType.T_L2,
            ErrorType.T_HDIV
        ]

        norms = {
            # Primal methods (Velocity in H^1)
            MethodType.TH: primal_norms,
            MethodType.MINI: primal_norms,
            MethodType.KS: primal_norms,
            
            # Dual-mixed / Stress-velocity methods (Stress in H(div), Velocity in L^2)
            MethodType.AFW: mixed_norms,
            MethodType.RTCG: mixed_norms,
            MethodType.HYBRID: mixed_norms,
            MethodType.AUG: mixed_norms,
        }
        return norms[self]