# states/state1.py
import ast
import numpy as np

def _validar_polinomio(expr: str) -> ast.AST:
    """
    Valida que 'expr' sea una expresión polinómica en x con operaciones seguras.
    Permitido: +, -, *, **, paréntesis, números y la variable x.
    No se permiten llamadas a funciones ni otros nombres/atributos.
    """
    tree = ast.parse(expr, mode="eval")

    allowed_nodes = (
        ast.Expression,
        ast.BinOp, ast.UnaryOp,
        ast.Add, ast.Sub, ast.Mult, ast.Pow,
        ast.UAdd, ast.USub,
        ast.Constant,  # números literales
        ast.Name,      # x
        ast.Load,
    )

    class Validator(ast.NodeVisitor):
        def visit_Call(self, node):
            raise ValueError("No se permiten llamadas a funciones (solo polinomios en x).")
        def visit_Attribute(self, node):
            raise ValueError("No se permiten atributos (solo polinomios en x).")
        def visit_Name(self, node):
            if node.id != "x":
                raise ValueError("Solo se permite la variable 'x'.")
        def generic_visit(self, node):
            if not isinstance(node, allowed_nodes):
                raise ValueError(f"Elemento no permitido en el polinomio: {type(node).__name__}")
            super().generic_visit(node)

    Validator().visit(tree)
    return tree

def _compilar_funcion(expr: str):
    """Compila la expresión validada a una función f(x) segura."""
    tree = _validar_polinomio(expr)
    code = compile(tree, filename="<polinomio>", mode="eval")

    def f(x):
        return eval(code, {"__builtins__": {}}, {"x": x})

    return f

def run():
    print("=== Estado 1: Ingreso de polinomio en x ===")
    print("Ejemplo: 3*x**2 - 2*x + 1")
    print("Notas: usá '**' para potencias, solo variable 'x' y operaciones +, -, *, **.\n")

    # Grid de evaluación
    x = np.linspace(-5, 5, 1000)

    # Tolerancias numéricas
    RTOL = 1e-8
    ATOL = 1e-10
    ATOL_MAG = 1e-12  # tolerancia para |f(x)| <= 1 en [-1, 1]

    while True:
        # 1) Leer, validar y compilar
        try:
            expr = input("Ingresá el polinomio en x: ").strip()
            f = _compilar_funcion(expr)
        except (SyntaxError, ValueError) as e:
            print(f"Error: {e}\nVolvé a intentarlo.\n")
            continue
        except (KeyboardInterrupt, EOFError):
            print("\nOperación cancelada por el usuario.")
            return

        # 2) Evaluar en [-5, 5]
        try:
            y = f(x)
            y_menos = f(-x)
        except Exception as e:
            print(f"Ocurrió un error al evaluar el polinomio: {e}\n")
            continue

        # 3) Chequeo par/impar en [-5, 5]
        es_par = np.allclose(y, y_menos, rtol=RTOL, atol=ATOL)
        es_impar = np.allclose(y, -y_menos, rtol=RTOL, atol=ATOL)

        if es_par:
            print("\nEl polinomio es PAR en el rango [-5, 5].")
        elif es_impar:
            print("\nEl polinomio es IMPAR en el rango [-5, 5].")
        else:
            err_par = float(np.max(np.abs(y - y_menos)))
            err_impar = float(np.max(np.abs(y + y_menos)))
            print(
                "\nEl polinomio NO es ni par ni impar en [-5, 5].\n"
                f"Desvío como par (max |f(x)-f(-x)|): {err_par:.3e}\n"
                f"Desvío como impar (max |f(x)+f(-x)|): {err_impar:.3e}\n"
                "Ingresá un nuevo polinomio.\n"
            )
            continue

        # 4) Chequeo de magnitud en [-1, 1]: |f(x)| <= 1
        mask = (x >= -1) & (x <= 1)
        y_sub = y[mask]
        max_abs = float(np.max(np.abs(y_sub)))

        if max_abs <= 1 + ATOL_MAG:
            y_min, y_max = float(np.min(y_sub)), float(np.max(y_sub))
            print(f"Magnitud verificada en [-1, 1]: max|f(x)| = {max_abs:.6g} (OK)")
            print(f"Resumen en [-1, 1]: y_min = {y_min:.6g}, y_max = {y_max:.6g}")
            print("Volviendo al menú...\n")
            break
        else:
            print(
                "\nEl polinomio NO cumple |f(x)| ≤ 1 en [-1, 1].\n"
                f"max|f(x)| en [-1, 1] = {max_abs:.6g} (> 1)\n"
                "Ingresá un nuevo polinomio.\n"
            )
            # vuelve a pedir otro polinomio