# states/state1.py
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ---------- Validación/compilación del polinomio ----------
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

# ---------- Utilidades seguras ----------
def _safe_inv_array(arr, eps=1e-12):
    z = np.empty_like(arr, dtype=float)
    mask = np.abs(arr) > eps
    z[mask] = 1.0 / arr[mask]
    z[~mask] = np.nan   # evita división por cero; NaN se ve como hueco
    if not np.all(mask):
        print("Aviso: hubo valores cercanos a 0; se muestran como NaN en 1/f(x).")
    return z

def _safe_inv_bounds(lo, hi, eps=1e-12):
    """Invierte de forma segura el intervalo [lo, hi] evitando 0 y devolviendo (lo', hi') ordenado."""
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

# ---------- Estado 1 ----------
def run():
    print("=== Estado 1: Ingreso de polinomio en x ===")
    print("Ejemplo: 3*x**2 - 2*x + 1")
    print("Notas: usá '**' para potencias, solo variable 'x' y operaciones +, -, *, **.\n")

    x = np.linspace(-5, 5, 10000)

    # Tolerancias
    RTOL = 1e-8
    ATOL = 1e-10
    ATOL_MAG = 1e-12  # para |f(x)| <= 1 en [-1, 1]

    # --------- Bucle de ingreso/validación ----------
    while True:
        try:
            expr = input("Ingresá el polinomio en x: ").strip()
            f = _compilar_funcion(expr)
        except (SyntaxError, ValueError) as e:
            print(f"Error: {e}\nVolvé a intentarlo.\n")
            continue
        except (KeyboardInterrupt, EOFError):
            print("\nOperación cancelada por el usuario.")
            return

        # Evaluación
        try:
            y = f(x)
            y_menos = f(-x)
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
        mask_11 = (x >= -1) & (x <= 1)
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
    (linea,) = ax.plot(x, y, label=r'$f(x)$', linewidth=1.5)
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
    # Inicial, luego por cada paso: Escalar, Cuadrado, Sumar, Invertir
    ylims_inicial = (-1.5, 7.0)
    ylims_por_paso = [
        (-1.5, 5.0),  # tras Escalar por R
        (-1.5, 3.0),  # tras Elevar al cuadrado
        (-1.5, 3.0),  # tras Sumar 1
        (0.0, 1.1),   # tras Invertir 1/f
    ]
    ax.set_ylim(*ylims_inicial)

    fig.canvas.draw()
    fig.canvas.flush_events()

    # --------- Secuencia de transformaciones por Enter ----------
    base_y = y.copy()
    y_work = base_y.copy()
    R = 1.0  # valor inicial para el escalado

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
            raw = input(f"Ingresá R (actual={R}, Enter para mantener): ").strip()
            if raw:
                try:
                    R = float(raw)
                except ValueError:
                    print("Valor inválido, se mantiene R anterior.")
            y_work = y_work * R
            rect_lo *= R
            rect_hi *= R
            etiqueta = f"(f) * {R:g}"

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

        # Actualizar curva y leyenda
        linea.set_ydata(y_work)
        linea.set_label(etiqueta)
        ax.legend(loc="best")

        # Actualizar rectángulo
        actualizar_rectangulo()

        # Aplicar Y-limits hardcodeados para este paso
        ymin, ymax = ylims_por_paso[i - 1]
        ax.set_ylim(ymin, ymax)

        # Redibujar (sin autoscale/relim)
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Enter final para cerrar
    try:
        input("Enter para cerrar la figura y volver al menú...")
    except (KeyboardInterrupt, EOFError):
        pass
    plt.close(fig)
    print("Volviendo al menú...\n")
