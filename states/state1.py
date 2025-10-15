# states/state1.py
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import signal

# ---------- SymPy para la expresión simbólica final ----------
import sympy as sp
from sympy.abc import x, xi, w, n, s  # incluimos w como variable simbólica objetivo
try:
    from IPython.display import display
    HAVE_DISPLAY = True
except Exception:
    HAVE_DISPLAY = False
sp.init_printing(use_unicode=True)

# ---------- Validación/compilación del polinomio (numérico) ----------
def _validar_polinomio(expr: str) -> ast.AST:
    tree = ast.parse(expr, mode="eval")
    allowed = (
        ast.Expression,
        ast.BinOp, ast.UnaryOp,
        ast.Add, ast.Sub, ast.Mult, ast.Pow,
        ast.UAdd, ast.USub,
        ast.Constant,
        ast.Name, ast.Load,
        ast.Tuple,
    )
    class V(ast.NodeVisitor):
        def visit_Call(self, node): raise ValueError("No se permiten llamadas a funciones (solo polinomios en x).")
        def visit_Attribute(self, node): raise ValueError("No se permiten atributos (solo polinomios en x).")
        def visit_Name(self, node):
            if node.id != "x":
                raise ValueError("Solo se permite la variable 'x'.")
        def generic_visit(self, node):
            if not isinstance(node, allowed):
                raise ValueError(f"Elemento no permitido: {type(node).__name__}")
            super().generic_visit(node)
    V().visit(tree)
    return tree

def _compilar_funcion(expr: str):
    code = compile(_validar_polinomio(expr), filename="<polinomio>", mode="eval")
    def f(x):
        return eval(code, {"__builtins__": {}}, {"x": x})
    return f

# ---------- Utilidades numéricas seguras ----------
def _safe_inv_array(arr, eps=1e-12):
    z = np.empty_like(arr, dtype=float)
    mask = np.abs(arr) > eps
    z[mask] = 1.0 / arr[mask]
    z[~mask] = np.nan   # evita división por cero; NaN se ve como hueco
    if not np.all(mask):
        print("Aviso: hubo valores cercanos a 0; se muestran como NaN en 1/f(x).")
    return z

def _safe_inv_bounds(lo, hi, eps=1e-12):
    """Invierte de forma segura el intervalo [lo, hi] evitando 0 y devuelve (lo', hi') ordenado."""
    if lo > hi:
        lo, hi = hi, lo
    def clip_away_from_zero(v):
        if -eps < v < eps:
            return eps if v >= 0 else -eps
        return v
    lo_c = clip_away_from_zero(lo)
    hi_c = clip_away_from_zero(hi)
    inv_lo = 1.0 / lo_c
    inv_hi = 1.0 / hi_c
    return (min(inv_lo, inv_hi), max(inv_lo, inv_hi))

def get_zeros_and_poles(F):
    """
    Función que calcula los ceros y los polos de una función transferencia

    :param F: función transferencia como expresión de sympy

    :returns: ceros y polos de la función transferencias
    """

    #Obtenemos  el numerador y denominador de la expresión
    num = sp.fraction(F)[0]
    den = sp.fraction(F)[1]


    #Obtenemos los coeficientes del numerador y denominador. Los ifs son para los casos donde el numerador o denominador son de orden 1.

    #si es de orden 1
    if not isinstance(num, sp.Poly):
        num_coeffs = num
    #Si es de orden mayor
    else:
        num_coeffs = sp.Poly(num).all_coeffs()

    if not isinstance(den, sp.Poly):
        den_coeffs = sp.Poly(den).all_coeffs()
    else:
        den_coeffs = den


    #Para mejor manejo vamos a pasar estos coeficientes a un tipo de arreglo de numpy (confíen)
    num_coeffs = np.array(num_coeffs, dtype = float)
    den_coeffs = np.array(den_coeffs, dtype = float)



    #Obtenemos las raíces de esos polinomios.
    #Esto lo vamos a hacer a través de scipy, que tiene una función que a partir de los coeficientes del numerador y denominador, te da los polos y ceros

    pz = signal.tf2zpk(num_coeffs,den_coeffs)

    #pz es una arreglo [ceros, polos, ganancia]

    zeros = pz[0]
    poles = pz[1]

    return zeros, poles

def plot_pz_map(zeros, poles):
    """
    Función que grafica el PZ map diagrama de polos y ceros.

    :param zeros: lista de ceros
    :param poles: lista de polos

    """
    plt.figure(figsize=(8, 6))
    zeros_parte_real = np.real(zeros)
    zeros_parte_imaginaria =  np.imag(zeros)
    plt.scatter(zeros_parte_real,zeros_parte_imaginaria, marker='o', color='b')

    poles_parte_real = np.real(poles)
    poles_parte_imaginaria =  np.imag(poles)
    plt.scatter(poles_parte_real,poles_parte_imaginaria, marker='x', color='b')

    #Pueden cambiar este límite como quieram
    plt.xlim([-1.6, 1.6])

    plt.xlabel('$\\sigma$')
    plt.ylabel('$j\\omega$')

    plt.grid()
    plt.show()


def plot_bode(mag, w, phase, phase_units="deg", unwrap=True):
    """
    Plotea el módulo y la fase de la respuesta en frecuencia (escala lineal).

    :param mag: array-like, magnitud |H(jw)| en escala lineal
    :param w: array-like, frecuencia (rad/s)
    :param phase: array-like, fase de H(jw)
    :param phase_units: 'deg' (default) o 'rad' — unidades en las que viene 'phase'
    :param unwrap: bool, si True intenta "desenvolver" (unwrap) la fase
    """
    mag = np.asarray(mag)
    w = np.asarray(w)
    phase = np.asarray(phase)

    if mag.shape != w.shape or phase.shape != w.shape:
        raise ValueError("mag, phase y w deben tener la misma forma.")

    # Unwrap de la fase
    if unwrap:
        if phase_units == "deg":
            phase_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(phase)))
        elif phase_units == "rad":
            phase_unwrapped = np.unwrap(phase)
        else:
            raise ValueError("phase_units debe ser 'deg' o 'rad'.")
    else:
        phase_unwrapped = phase

    # Etiqueta de la fase según unidades
    phase_label = "Fase [deg]" if phase_units == "deg" else "Fase [rad]"

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    # Magnitud (lineal)
    ax_mag.plot(w, mag)
    ax_mag.set_ylabel('Módulo [v/v]')
    ax_mag.grid(True, which='both')
    ax_mag.set_title('Respuesta en frecuencia (lineal)')

    # Fase
    ax_phase.plot(w, phase_unwrapped)
    ax_phase.set_ylabel(phase_label)
    ax_phase.set_xlabel(r'$\omega$  [rad/s]')
    ax_phase.grid(True, which='both')

    plt.tight_layout()
    plt.show()





# ---------- Estado 1 ----------
def run():
    print("=== Estado 1: Ingreso de polinomio en x ===")
    print("Ejemplo: 3*x**2 - 2*x + 1")
    print("Notas: usá '**' para potencias, solo variable 'x' y operaciones +, -, *, **.\n")

    x_np = np.linspace(-5, 5, 1000)

    # Tolerancias
    RTOL = 1e-8
    ATOL = 1e-10
    ATOL_MAG = 1e-12  # para |f(x)| <= 1 en [-1, 1]

    # --------- Bucle de ingreso/validación ----------
    while True:
        try:
            expr = input("Ingresá el polinomio en x: ").strip()
            f_num = _compilar_funcion(expr)  # función NUMÉRICA
        except (SyntaxError, ValueError) as e:
            print(f"Error: {e}\nVolvé a intentarlo.\n")
            continue
        except (KeyboardInterrupt, EOFError):
            print("\nOperación cancelada por el usuario.")
            return

        # Evaluación numérica
        try:
            y = f_num(x_np)
            y_menos = f_num(-x_np)
        except Exception as e:
            print(f"Ocurrió un error al evaluar el polinomio: {e}\n")
            continue

        # Paridad en [-5,5]
        es_par = np.allclose(y, y_menos, rtol=RTOL, atol=ATOL)
        es_impar = np.allclose(y, -y_menos, rtol=RTOL, atol=ATOL)
        if not (es_par or es_impar):
            err_par = float(np.max(np.abs(y - y_menos)))
            err_impar = float(np.max(np.abs(y + y_menos)))
            print(
                "\nEl polinomio NO es ni par ni impar en [-5, 5].\n"
                f"Desvío como par (max |f(x)-f(-x)|): {err_par:.3e}\n"
                f"Desvío como impar (max |f(x)+f(-x)|): {err_impar:.3e}\n"
                "Ingresá un nuevo polinomio.\n"
            )
            continue
        print("\nEl polinomio es PAR." if es_par else "\nEl polinomio es IMPAR.")

        # Magnitud en [-1,1]
        mask_11 = (x_np >= -1) & (x_np <= 1)
        y_sub = y[mask_11]
        max_abs = float(np.max(np.abs(y_sub)))
        if max_abs > 1 + ATOL_MAG:
            print(
                "\nEl polinomio NO cumple |f(x)| ≤ 1 en [-1, 1].\n"
                f"max|f(x)| en [-1, 1] = {max_abs:.6g} (> 1)\n"
                "Ingresá un nuevo polinomio.\n"
            )
            continue

        # Si pasó todos los chequeos, salimos del bucle
        break

    # --------- Gráfico inicial ---------
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 5))
    (linea,) = ax.plot(x_np, y, label=r'$f(x)$', linewidth=1.5)
    ax.plot((0, 3.5), (0, 0), color="black", linewidth=0.5)

    # Rectángulo: x fijo en [0,1], y inicial en [-1,1] (seguirá las transformaciones)
    x0, width = 0.0, 1.0
    rect_lo, rect_hi = -1.0, 1.0
    rect = Rectangle((x0, rect_lo), width, rect_hi - rect_lo, facecolor='green', alpha=0.2)
    ax.add_patch(rect)

    ax.set_xlabel('x')
    ax.set_xlim([0, 2.5])
    ax.legend(loc="best")

    # Y-limits hardcodeados por etapa:
    ylims_inicial = (-1.5, 7.0)
    ylims_por_paso = [
        (-1.5, 5.0),  # tras Escalar por R
        (-1.5, 3.0),  # tras Elevar al cuadrado
        (-1.5, 3.0),  # tras Sumar 1
        (0.0, 1.2),   # tras Invertir 1/f
    ]
    ax.set_ylim(*ylims_inicial)

    fig.canvas.draw()
    fig.canvas.flush_events()

    # --------- Secuencia de transformaciones por Enter (NUMÉRICAS) ----------
    base_y = y.copy()
    y_work = base_y.copy()
    R_value = 1.0  # valor numérico inicial para el escalado

    pasos = [
        "Escalar por R",
        "Elevar al cuadrado (f → f^2)",
        "Sumar 1 (f → f + 1)",
        "Invertir (f → 1/f)",
    ]

    def actualizar_rectangulo():
        lo, hi = (rect_lo, rect_hi) if rect_lo <= rect_hi else (rect_hi, rect_lo)
        rect.set_xy((x0, lo))
        rect.set_width(width)
        rect.set_height(hi - lo)

    for i, paso in enumerate(pasos, start=1):
        try:
            input(f"Enter para aplicar: {paso} (paso {i}/{len(pasos)}). Ctrl+C para salir...")
        except (KeyboardInterrupt, EOFError):
            print("\nCerrando figura y volviendo al menú...")
            plt.close(fig)
            return

        if paso == "Escalar por R":
            raw = input(f"Ingresá R (actual={R_value}, Enter para mantener): ").strip()
            if raw:
                try:
                    R_value = float(raw)
                except ValueError:
                    print("Valor inválido, se mantiene R anterior.")
            y_work = y_work * R_value
            rect_lo *= R_value
            rect_hi *= R_value
            etiqueta = f"(f) * {R_value:g}"

        elif paso == "Elevar al cuadrado (f → f^2)":
            y_work = y_work ** 2
            # Rectángulo: borde inferior a 0 y superior = max(|lo|,|hi|)^2
            max_abs_edge = max(abs(rect_lo), abs(rect_hi))
            rect_lo = 0.0
            rect_hi = max_abs_edge ** 2
            etiqueta = "(f)^2"

        elif paso == "Sumar 1 (f → f + 1)":
            y_work = y_work + 1.0
            rect_lo += 1.0
            rect_hi += 1.0
            etiqueta = "f + 1"

        elif paso == "Invertir (f → 1/f)":
            y_work = _safe_inv_array(y_work)
            rect_lo, rect_hi = _safe_inv_bounds(rect_lo, rect_hi)
            etiqueta = "1 / f"

        # Actualizar curva y rectángulo
        linea.set_ydata(y_work)
        linea.set_label(etiqueta)
        ax.legend(loc="best")
        actualizar_rectangulo()

        # Aplicar Y-limits hardcodeados para este paso
        ymin, ymax = ylims_por_paso[i - 1]
        ax.set_ylim(ymin, ymax)

        fig.canvas.draw()
        fig.canvas.flush_events()

    # --------- Construcción simbólica final G(w) (NO NUMPY) ----------
    # 1) Construir f(w) simbólica a partir del string del usuario:
    #    mapeamos la 'x' de su texto a 'w' para que la variable de la expresión sea w.
    try:
        f_sym_w = sp.sympify(expr, locals={'x': w, 'w': w, 'xi': xi, 'n': n, 's': s})
    except Exception as e:
        print(f"\n(No se pudo construir simbólicamente el polinomio ingresado: {e})")
        f_sym_w = sp.Symbol('f')  # fallback simbólico

    # 2) R simbólica
    R_sym = sp.Symbol('R', real=True)

    # 3) Replicar las transformaciones simbólicas con variable w:
    #    f -> R*f -> (R*f)^2 -> (R*f)^2 + 1 -> 1 / ((R*f)^2 + 1)
    G = 1 / ((R_sym * f_sym_w)**2 + 1)   # <-- G(w)

    G_num = G.subs(R_sym, sp.nsimplify(R_value))
    print("\nG(w)= 1 / ((R*f(w))**2 + 1):")
    if HAVE_DISPLAY:
        display(G_num)
    else:
        sp.pprint(G)

    # 4) Sustitución w -> s / i
    Gs = G_num.subs(w, s / sp.I)
    print("\nG(s/i) = G(w -> s/i):")
    if HAVE_DISPLAY:
        display(Gs)
    else:
        sp.pprint(Gs)

    # Enter final para cerrar
    try:
        input("\nEnter para cerrar la figura y volver al menú...")
    except (KeyboardInterrupt, EOFError):
        pass
    plt.close(fig)

    zeros, poles = get_zeros_and_poles(Gs)

    useful_poles = [p for p in poles if np.real(p) < 0]

    print('ceros: ', list(zeros))
    print('polos del lado izquierdo: ', list(useful_poles))

    plot_pz_map(zeros,useful_poles)

    zeros = zeros
    poles =  useful_poles
    gain = np.cumprod(useful_poles)[-1]

    print('La ganancia es: ', gain)

    #Define un sistema de scipy a través de los ceros, polos y ganancia
    tf = signal.ZerosPolesGain(zeros, poles, gain )

    w_rad = np.linspace(0,2,100)

    w_rad, mag, phase = signal.bode(tf, w = w_rad)
    mag_en_veces = 10**(mag/20)
    plot_bode(mag_en_veces, w_rad, phase)


    print("Volviendo al menú...\n")
