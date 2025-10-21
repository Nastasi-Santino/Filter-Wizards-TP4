# utils/pz_tools.py
import numpy as np
import sympy as sp

try:
    from scipy import signal
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

import matplotlib.pyplot as plt

# -------------------------
#   Zeros & Poles (SymPy)
# -------------------------
def get_zeros_and_poles(F, var=sp.symbols('s'), params=None, use_scipy=False):
    """
    Calcula ceros y polos de una función transferencia F(var).

    F        : expresión SymPy racional en 'var'
    var      : símbolo de variable (por ej., s)
    params   : dict para sustituir parámetros simbólicos por valores numéricos, ej {R: 0.5}
    use_scipy: si True usa scipy.signal.tf2zpk (requiere coeficientes numéricos)

    return: (zeros, poles) como arrays de complex
    """
    # Normalizar a fracción y sustituir parámetros si los hay
    F = sp.together(F)
    num, den = sp.fraction(F)  # Expr, Expr

    # Detectar símbolos “no var”
    other_syms = (num.free_symbols | den.free_symbols) - {var}
    if other_syms:
        if not params:
            raise ValueError(
                f"La expresión contiene símbolos {other_syms}. "
                f"Pasá 'params={{...}}' con valores numéricos (p.ej. {{R: 0.5}})."
            )
        num = sp.N(num.subs(params))
        den = sp.N(den.subs(params))

    # Convertir a Poly en la variable
    num = sp.expand(num)
    den = sp.expand(den)
    try:
        num_poly = sp.Poly(num, var)
    except sp.PolynomialError:
        raise ValueError(f"El numerador no es polinómico en {var}: {num}")
    try:
        den_poly = sp.Poly(den, var)
    except sp.PolynomialError:
        raise ValueError(f"El denominador no es polinómico en {var}: {den}")

    deg_num = num_poly.degree()
    deg_den = den_poly.degree()

    if not use_scipy:
        # Usamos SymPy (nroots) → no requiere SciPy
        zeros = []
        if deg_num > 0:
            zeros = [complex(r) for r in sp.nroots(num_poly)]
        poles = []
        if deg_den > 0:
            poles = [complex(r) for r in sp.nroots(den_poly)]
        return np.array(zeros, dtype=complex), np.array(poles, dtype=complex)

    # SciPy: tf2zpk con coeficientes numéricos
    if not HAVE_SCIPY:
        raise RuntimeError("SciPy no disponible. Usá use_scipy=False o instalá scipy.")

    def coeffs_to_np(poly):
        return np.array([complex(sp.N(c)) for c in poly.all_coeffs()], dtype=complex)

    b = coeffs_to_np(num_poly) if deg_num >= 0 else np.array([0.0], dtype=complex)
    a = coeffs_to_np(den_poly)
    z, p, k = signal.tf2zpk(b, a)
    return np.array(z, dtype=complex), np.array(p, dtype=complex)


# -------------------------
#   PZ map con semicírculo
# -------------------------
def plot_pz_map(zeros, poles, r=1.0, fill=False):
    """
    Grafica mapa de polos y ceros con semicírculo de radio r en el semiplano izquierdo.

    zeros, poles: arrays de números complejos
    r           : radio del semicírculo (default 1.0)
    fill        : si True, rellena el semicírculo con transparencia
    """
    plt.figure(figsize=(8, 6))

    # Puntos
    plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='b', label='Ceros')
    plt.scatter(np.real(poles), np.imag(poles), marker='x', color='r', label='Polos')

    # Ejes
    plt.axhline(0, color='k', linewidth=0.8)
    plt.axvline(0, color='k', linewidth=0.8)

    # Semicírculo: semiplano izquierdo (theta de +pi/2 a +3pi/2)
    theta = np.linspace(np.pi/2, 3*np.pi/2, 400)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    plt.plot(x, y, 'g--', linewidth=1.5, label=f'Semicírculo r={r}')
    if fill:
        plt.fill_betweenx(y, x, 0, where=(x <= 0), color='g', alpha=0.12)

    # Límites y aspecto
    # Auto-límites cómodos según datos y r
    all_x = np.concatenate([np.real(zeros), np.real(poles), x])
    all_y = np.concatenate([np.imag(zeros), np.imag(poles), y])
    if all_x.size and all_y.size:
        dx = max(1.2 * np.max(np.abs(all_x)), r * 1.2, 1.6)
        dy = max(1.2 * np.max(np.abs(all_y)), r * 1.2, 1.6)
    else:
        dx = dy = max(r * 1.2, 1.6)
    plt.xlim(-dx, dx)
    plt.ylim(-dy, dy)

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.xlabel(r'$\sigma$')
    plt.ylabel(r'$j\omega$')
    plt.grid(True)
    plt.legend(loc='best')
    plt.title('Mapa de Polos y Ceros')
    plt.show()







def reverse_bessel_poly(n: int, var=None):
    """
    Devuelve (Theta_n(s), s), el polinomio de Bessel 'reverso' de orden n.
    Recurrencia:
        Θ_0(s) = 1
        Θ_1(s) = s + 1
        Θ_{m+1}(s) = (2m+1) * s * Θ_m(s) + Θ_{m-1}(s),   m >= 1
    """
    if var is None:
        var = s

    if n <= 0:
        return sp.Integer(1), var
    if n == 1:
        return var + 1, var

    theta_prev2 = sp.Integer(1)    # Θ_0
    theta_prev1 = var + 1          # Θ_1
    for m in range(1, n):          # genera Θ_2 ... Θ_n
        theta = (2*m + 1) * var * theta_prev1 + theta_prev2
        theta_prev2, theta_prev1 = theta_prev1, sp.expand(theta)
    return theta_prev1, var
