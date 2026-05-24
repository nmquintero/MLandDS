"""
╔══════════════════════════════════════════════════════════════╗
║  TASK 3 — Inventory Management (Linear Programming)         ║
║  Gestión de inventario con programación lineal              ║
║                                                             ║
║  Algoritmo: Linear Programming (PuLP + solver CBC)          ║
║  Concepto: modelar el problema matemáticamente y            ║
║            dejar que un solver encuentre el óptimo          ║
╚══════════════════════════════════════════════════════════════╝

PROBLEMA:
  Una empresa debe gestionar su inventario a lo largo de varios
  períodos (ej: meses). Sabe la demanda futura y necesita decidir
  cuánto pedir en cada período para:
    - Satisfacer toda la demanda (no tener faltantes)
    - No exceder la capacidad de almacén
    - Minimizar el costo total (pedido + almacenamiento)

  DIFERENCIA con Greedy y Brute Force:
    - Greedy: decisiones locales, rápido, subóptimo
    - Brute Force: prueba TODO, óptimo pero inviable para problemas grandes
    - LP: MODELAMOS el problema matemáticamente y el solver
           encuentra el óptimo de forma INTELIGENTE

  La programación lineal busca el punto óptimo en un "espacio de
  soluciones" definido por restricciones lineales. Es como encontrar
  el punto más bajo de un valle, donde las paredes del valle son
  las restricciones del problema.

MODELO MATEMÁTICO:

  Variables de decisión (lo que el solver va a calcular):
    x[t] = cantidad a pedir en el período t  (t = 0, 1, ..., T-1)
    I[t] = inventario al final del período t

  Parámetros (datos conocidos):
    D[t] = demanda en el período t
    C_ordenar = costo por unidad pedida
    C_almacenar = costo por unidad almacenada por período
    capacidad_max = capacidad máxima del almacén
    T = número de períodos

  Función objetivo (lo que queremos minimizar):
    Minimizar: Σ (C_ordenar * x[t] + C_almacenar * I[t])
               para todo t = 0, ..., T-1

    Esto significa: por cada período, pagamos por lo que pedimos
    más por lo que guardamos. Queremos el balance óptimo.

  Restricciones (reglas que se deben cumplir):

    1. Balance de inventario:
         I[t] = I[t-1] + x[t] - D[t]    ∀ t
       → Lo que tenemos al final = lo que había + lo que pedimos - lo que vendimos

    2. No negatividad del inventario:
         I[t] >= 0    ∀ t
       → No podemos tener inventario negativo (significaría faltantes)

    3. Capacidad máxima:
         I[t] <= capacidad_max    ∀ t
       → No podemos almacenar más de lo que entra en el depósito

    4. No negatividad de pedidos:
         x[t] >= 0    ∀ t
       → No podemos pedir cantidades negativas

    5. Inventario inicial:
         I[0] = 0
       → Empezamos con el almacén vacío

  Por qué funciona:
    - Todas las ecuaciones son LINEALES (x e I aparecen sin exponentes)
    - Las variables son CONTINUAS (pueden tomar cualquier valor real ≥ 0)
    - Los solvers como CBC usan el algoritmo SIMPLEX para encontrar
      el óptimo global de forma eficiente
    - Para problemas lineales, el óptimo global está GARANTIZADO

COMPLEJIDAD:
  - El algoritmo Simplex es O(n²) en la práctica, O(2ⁿ) en el peor caso teórico
  - PuLP usa CBC (COIN-OR Branch and Cut) que resuelve problemas con
    miles de variables en segundos
  - Mucho más eficiente que fuerza bruta para este tipo de problemas
"""

import pulp
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────
# DATOS DE ENTRADA
# ─────────────────────────────────────────────────────────────

# Demanda estimada para cada período (ej: unidades por mes)
# T = 12 períodos (un año completo)
demanda: List[float] = [
    100,   # Enero
    120,   # Febrero
    150,   # Marzo
    130,   # Abril
    160,   # Mayo
    200,   # Junio
    180,   # Julio
    170,   # Agosto
    140,   # Septiembre
    110,   # Octubre
    90,    # Noviembre
    80,    # Diciembre
]

# Costos
costo_ordenar: float = 5.0    # Costo por unidad pedida ($/unidad)
costo_almacenar: float = 2.0  # Costo por unidad almacenada por período ($/unidad/mes)
capacidad_max: float = 300.0   # Capacidad máxima del almacén (unidades)

T = len(demanda)  # Número de períodos


# ─────────────────────────────────────────────────────────────
# IMPLEMENTACIÓN DEL MODELO DE PROGRAMACIÓN LINEAL
# ─────────────────────────────────────────────────────────────

def inventory_lp(
    demanda: List[float],
    costo_ordenar: float,
    costo_almacenar: float,
    capacidad_max: float
) -> Tuple[pulp.LpProblem, List[pulp.LpVariable], List[pulp.LpVariable], float]:
    """
    Resuelve el problema de gestión de inventario usando programación
    lineal con PuLP y el solver CBC.

    Parámetros:
        demanda: lista con la demanda de cada período
        costo_ordenar: costo por unidad pedida
        costo_almacenar: costo por unidad almacenada por período
        capacidad_max: capacidad máxima del almacén

    Retorna:
        (problema, variables_x, variables_I, costo_total)
            problema: el modelo PuLP resuelto
            variables_x: lista de variables de decisión (pedidos)
            variables_I: lista de variables (inventario)
            costo_total: valor óptimo de la función objetivo

    Cómo construir un modelo de LP (paso a paso):

    PASO 1: Crear el problema
        → pulp.LpProblem("nombre", pulp.LpMinimize)
        → El segundo argumento es el sentido de la optimización:
           LpMinimize = minimizar, LpMaximize = maximizar

    PASO 2: Definir las variables de decisión
        → pulp.LpVariable("nombre", lowBound=0, upBound=..., cat="Continuous")
        → lowBound=0 significa x >= 0 (no negatividad)
        → cat="Continuous" significa variable continua (vs Integer o Binary)

    PASO 3: Definir la función objetivo
        → problema += expresión, "nombre_explicativo"
        → Ej: problema += 5*x1 + 2*x2, "Costo total"

    PASO 4: Agregar las restricciones
        → problema += expresión_relacional, "nombre_restriccion"
        → Ej: problema += x1 + x2 <= 100, "Capacidad_max"

    PASO 5: Resolver
        → problema.solve()
        → PuLP usa el solver CBC por defecto

    PASO 6: Obtener resultados
        → pulp.value(variable) → valor óptimo de esa variable
        → pulp.value(problema.objective) → valor óptimo de la función objetivo
    """

    # ─── PASO 1: Crear el problema ────────────────────────
    # Creamos un problema de minimización
    problema = pulp.LpProblem("Gestion_de_Inventario", pulp.LpMinimize)

    # ─── PASO 2: Definir las variables de decisión ────────
    # x[t] = cantidad a pedir en el período t (>= 0, continua)
    # I[t] = inventario al final del período t (>= 0, continua, <= capacidad_max)
    x = [
        pulp.LpVariable(f"x_{t}", lowBound=0, cat="Continuous")
        for t in range(T)
    ]
    I = [
        pulp.LpVariable(
            f"I_{t}",
            lowBound=0,
            upBound=capacidad_max,  # No podemos exceder la capacidad
            cat="Continuous"
        )
        for t in range(T)
    ]

    # ─── PASO 3: Función objetivo ─────────────────────────
    # Minimizar: Σ (costo_ordenar * x[t] + costo_almacenar * I[t])
    # Esto representa el costo total: pedir + guardar
    costo_total_expr = pulp.lpSum([
        costo_ordenar * x[t] + costo_almacenar * I[t]
        for t in range(T)
    ])
    problema += costo_total_expr, "Costo_Total"

    # ─── PASO 4: Restricciones ────────────────────────────

    # Restricción 1: Balance de inventario
    #   I[t] = I[t-1] + x[t] - D[t]    ∀ t
    #   Es decir: inventario_final = inventario_inicial + pedido - demanda
    for t in range(T):
        if t == 0:
            # Primer período: I[0] = 0 + x[0] - D[0]
            # I[0] = x[0] - D[0]  →  I[0] - x[0] = -D[0]
            problema += I[t] == x[t] - demanda[t], f"Balance_{t}"
        else:
            # Períodos siguientes: I[t] = I[t-1] + x[t] - D[t]
            # I[t] - I[t-1] - x[t] = -D[t]
            problema += I[t] == I[t-1] + x[t] - demanda[t], f"Balance_{t}"

    # Restricción 4: Capacidad máxima (ya está en upBound de las variables I)
    # No hace falta agregarla explícitamente porque la definimos al crear I[t]

    # ─── PASO 5: Resolver ─────────────────────────────────
    # solve() ejecuta el solver CBC (viene incluido con PuLP)
    # El solver aplica el algoritmo Simplex para encontrar el óptimo
    problema.solve(pulp.PULP_CBC_CMD(msg=False))  # msg=False para no mostrar logs del solver

    # ─── PASO 6: Obtener el costo total óptimo ────────────
    costo_total = pulp.value(problema.objective)

    return problema, x, I, costo_total


# ─────────────────────────────────────────────────────────────
# FUNCIÓN PARA MOSTRAR RESULTADOS
# ─────────────────────────────────────────────────────────────

def mostrar_resultados(
    problema: pulp.LpProblem,
    x: List[pulp.LpVariable],
    I: List[pulp.LpVariable],
    costo_total: float
) -> None:
    """
    Muestra la solución óptima encontrada por el solver.

    Formato:
        Período  | Demanda | Pedido | Inventario
        -----------------------------------------
        1 (Ene)  |   100   |   120  |     20
        ...

    También muestra el estado del solver y la interpretación económica.
    """

    print()
    print("╔════════════════════════════════════════════════════╗")
    print("║   PLAN ÓPTIMO DE INVENTARIO (Programación Lineal) ║")
    print("╚════════════════════════════════════════════════════╝")
    print()

    # ─── Estado del solver ───────────────────────────────
    # pulp.LpStatusOptimal = 1 (solución óptima encontrada)
    # Otros estados: Infeasible (sin solución), Unbounded (sin cota)
    print(f"  📊 Estado del solver: {pulp.LpStatus[problema.status]}")
    print()
    print(f"  💰 Costo total óptimo: ${costo_total:,.2f}")
    print()

    # ─── Tabla de resultados ─────────────────────────────
    nombres_meses = [
        "Ene", "Feb", "Mar", "Abr", "May", "Jun",
        "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"
    ]

    print("  ┌──────────┬─────────┬─────────┬────────────┐")
    print("  │ Período  │ Demanda │ Pedido  │ Inventario │")
    print("  ├──────────┼─────────┼─────────┼────────────┤")

    for t in range(T):
        mes = nombres_meses[t] if t < len(nombres_meses) else f"Período {t+1}"
        valor_x = pulp.value(x[t])
        valor_I = pulp.value(I[t])

        # Formateamos con ancho fijo para alinear columnas
        print(f"  │ {mes:8s} │ {demanda[t]:6.0f}  │ {valor_x:6.1f} │ {valor_I:9.1f} │")

    print("  └──────────┴─────────┴─────────┴────────────┘")
    print()

    # ─── Análisis de costos ──────────────────────────────
    costo_pedidos = sum(pulp.value(x[t]) * costo_ordenar for t in range(T))
    costo_almacenamiento = sum(pulp.value(I[t]) * costo_almacenar for t in range(T))

    print(f"  📋 Desglose de costos:")
    print(f"      Costo de pedidos:       ${costo_pedidos:,.2f}")
    print(f"      Costo de almacenamiento: ${costo_almacenamiento:,.2f}")
    print(f"      ─────────────────────────────────────")
    print(f"      Costo TOTAL:             ${costo_total:,.2f}")
    print()

    # ─── Interpretación ──────────────────────────────────
    print("  💡 Interpretación:")
    print("      El solver encontró la cantidad óptima a pedir en")
    print("      cada período para minimizar el costo total,")
    print("      respetando la capacidad del almacén y")
    print("      asegurando que nunca falte inventario.")
    print()


# ─────────────────────────────────────────────────────────────
# EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔════════════════════════════════════════════════════╗")
    print("║  TASK 3 — Inventory Management (Linear Prog.)     ║")
    print("╚════════════════════════════════════════════════════╝")
    print()
    print(f"  📦  Períodos: {T} meses")
    print(f"  📈  Demanda total: {sum(demanda):.0f} unidades")
    print(f"  💵  Costo por pedido: ${costo_ordenar:.2f}/unidad")
    print(f"  🏪  Costo almacenamiento: ${costo_almacenar:.2f}/unidad/mes")
    print(f"  📐  Capacidad máxima: {capacidad_max:.0f} unidades")
    print()

    # ─── Resolvemos el modelo ────────────────────────────
    problema, x, I, costo_total = inventory_lp(
        demanda, costo_ordenar, costo_almacenar, capacidad_max
    )

    # ─── Mostramos los resultados ────────────────────────
    mostrar_resultados(problema, x, I, costo_total)

    # ─── Análisis de complejidad ──────────────────────────
    print("╔════════════════════════════════════════════════════╗")
    print("║           ANÁLISIS DE COMPLEJIDAD                  ║")
    print("╚════════════════════════════════════════════════════╝")
    print()
    print("  Algoritmo: Simplex (Programación Lineal)")
    print("  Solver: CBC (COIN-OR Branch and Cut)")
    print("  Complejidad: O(n²) en la práctica")
    print()
    print(f"  Variables de decisión: {T} (pedidos) + {T} (inventario) = {2*T}")
    print(f"  Restricciones: {T} (balance) + 2*{T} (cotas) = {3*T}")
    print()
    print("  📘  Explicación:")
    print("       - A diferencia del greedy, NO toma decisiones locales")
    print("       - A diferencia de brute force, NO enumera opciones")
    print("       - CONSTRUYE un modelo matemático del problema")
    print("       - El solver navega el espacio de soluciones de forma")
    print("         inteligente usando el algoritmo Simplex")
    print("       - Garantiza el ÓPTIMO GLOBAL para problemas lineales")
    print("       - Escala a miles/millones de variables")
    print()
    print("  ⚖️  Comparación de los 3 enfoques:")
    print()
    print("  ┌────────────────┬──────────┬──────────┬──────────┐")
    print("  │                │  Greedy  │ Brute F. │    LP    │")
    print("  ├────────────────┼──────────┼──────────┼──────────┤")
    print("  │ Óptimo global? │    No    │    Sí    │    Sí    │")
    print("  │ Velocidad      │  Rápido  │  Lento   │  Rápido  │")
    print("  │ Escalabilidad  │  Alta    │  Baja    │  Alta    │")
    print("  │ Completitud    │  Parcial │  Total   │  Total   │")
    print("  │ Dif. impl.     │  Baja    │  Baja    │  Media   │")
    print("  └────────────────┴──────────┴──────────┴──────────┘")
    print()
