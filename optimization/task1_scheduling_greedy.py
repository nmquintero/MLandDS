"""
╔══════════════════════════════════════════════════════════════╗
║  TASK 1 — Resource Scheduling (Greedy)                      ║
║  Asignación de tareas a recursos con capacidad limitada     ║
║                                                             ║
║  Algoritmo: Greedy (shortest-task-first)                    ║
║  Concepto: decisiones localmente óptimas para construir     ║
║            una solución factible paso a paso                ║
╚══════════════════════════════════════════════════════════════╝

PROBLEMA:
  Tenemos N tareas. Cada tarea tiene:
    - duración: tiempo que tarda en ejecutarse
    - recursos_necesarios: lista de recursos que necesita

  Tenemos M recursos. Cada recurso tiene:
    - nombre: identificador
    - capacidad: cuántas unidades de trabajo puede manejar

  Objetivo: asignar tareas a recursos respetando:
    - capacidad de cada recurso (no se puede exceder)
    - deadline global (todas las tareas deben terminar antes)

ALGORITMO GREEDY:
  1. Ordenar tareas por duración (más cortas primero)
     → Las tareas cortas son más fáciles de "encajar"
  2. Para cada tarea (de la más corta a la más larga):
     a. Buscar el primer recurso con capacidad suficiente
     b. Si se encuentra y no excede el deadline → asignar
     c. Si no → la tarea queda sin asignar
  3. Mostrar el plan resultante

COMPLEJIDAD: O(N * M) donde N = tareas, M = recursos
"""

from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────────────────────
# DATOS DE ENTRADA
# ─────────────────────────────────────────────────────────────

# Definimos las tareas con su duración (en horas) y qué recursos necesitan.
# Cada tarea es una tupla: (nombre, duracion, [recursos_necesarios])

tareas: List[Tuple[str, int, List[str]]] = [
    ("Tarea A", 3, ["CPU", "RAM"]),    # Necesita CPU y RAM
    ("Tarea B", 1, ["CPU"]),           # Solo necesita CPU
    ("Tarea C", 5, ["GPU", "RAM"]),    # Necesita GPU y RAM
    ("Tarea D", 2, ["CPU", "GPU"]),    # Necesita CPU y GPU
    ("Tarea E", 4, ["RAM"]),           # Solo necesita RAM
    ("Tarea F", 2, ["CPU", "RAM"]),    # Necesita CPU y RAM
]

# Definimos los recursos disponibles.
# Cada recurso es un diccionario con su nombre y capacidad máxima.

recursos: List[Dict] = [
    {"nombre": "CPU", "capacidad": 6},   # Puede tomar hasta 6 unidades de trabajo
    {"nombre": "GPU", "capacidad": 4},   # Puede tomar hasta 4 unidades
    {"nombre": "RAM", "capacidad": 5},   # Puede tomar hasta 5 unidades
]

# Deadline global: todas las tareas deben completarse antes de este tiempo
deadline: int = 15  # horas


# ─────────────────────────────────────────────────────────────
# IMPLEMENTACIÓN DEL ALGORITMO GREEDY
# ─────────────────────────────────────────────────────────────

def resource_scheduling_greedy(
    tareas: List[Tuple[str, int, List[str]]],
    recursos: List[Dict],
    deadline: int
) -> Tuple[Dict, int]:
    """
    Algoritmo greedy para asignar tareas a recursos.

    Parámetros:
        tareas: lista de (nombre, duracion, [recursos_necesarios])
        recursos: lista de {"nombre": str, "capacidad": int}
        deadline: tiempo máximo para completar todas las tareas

    Retorna:
        (plan, tiempo_total):
            plan = {nombre_tarea: (recurso_asignado, inicio, fin)}
            tiempo_total = suma de duraciones de tareas asignadas

    Cómo funciona:
        1. Ordenamos las tareas por duración (ascendente)
           → Las tareas más cortas primero porque son más fáciles
           de ubicar en los recursos disponibles
        2. Inicializamos el "tiempo ocupado" de cada recurso en 0
        3. Para cada tarea (ya ordenada):
           a. Buscamos un recurso que tenga capacidad suficiente
           b. Verificamos que el recurso esté disponible (no exceda deadline)
           c. Si cumple ambas condiciones → asignamos la tarea
           d. Actualizamos el tiempo del recurso
        4. Retornamos el plan y el tiempo total

    Por qué funciona:
        - Es rápido (O(N * M))
        - Da una solución factible aunque no necesariamente óptima global
        - Es útil cuando necesitamos una respuesta rápida sin garantía de optimalidad
    """

    # 1. Ordenar tareas por duración (menor a mayor)
    #    sorted() con key=lambda x: x[1] ordena por el segundo elemento (duracion)
    tareas_ordenadas = sorted(tareas, key=lambda t: t[1])

    # 2. Estado de cada recurso: cuánto tiempo lleva ocupado
    #    Inicializamos cada recurso con tiempo 0 (todos disponibles)
    tiempo_recurso: Dict[str, int] = {r["nombre"]: 0 for r in recursos}

    # 3. Plan de asignación: {nombre_tarea: (recurso, inicio, fin)}
    plan: Dict[str, Tuple[str, int, int]] = {}
    tareas_no_asignadas: List[str] = []

    # 4. Iteramos sobre cada tarea (de la más corta a la más larga)
    for nombre_tarea, duracion, recursos_necesarios in tareas_ordenadas:
        asignada = False

        # Intentamos asignar la tarea a algún recurso
        for recurso in recursos:
            nombre_recurso = recurso["nombre"]
            capacidad_max = recurso["capacidad"]

            # Verificamos si este recurso es uno de los necesitados
            if nombre_recurso not in recursos_necesarios:
                continue

            # Verificamos si el recurso tiene capacidad disponible
            # capacidad_max - tiempo_recurso[nombre_recurso] = capacidad restante
            if tiempo_recurso[nombre_recurso] + duracion > capacidad_max:
                continue

            # Verificamos si la tarea cabe dentro del deadline
            if tiempo_recurso[nombre_recurso] + duracion > deadline:
                continue

            # Si pasamos todas las verificaciones → ASIGNAMOS la tarea
            inicio = tiempo_recurso[nombre_recurso]
            fin = inicio + duracion

            # Guardamos la asignación en el plan
            plan[nombre_tarea] = (nombre_recurso, inicio, fin)

            # Actualizamos el tiempo ocupado del recurso
            tiempo_recurso[nombre_recurso] = fin

            asignada = True
            break  # Salimos del bucle de recursos, pasamos a la siguiente tarea

        # Si no se pudo asignar, la registramos como no asignada
        if not asignada:
            tareas_no_asignadas.append(nombre_tarea)

    # 5. Calculamos el tiempo total (suma de duraciones asignadas)
    tiempo_total = sum(t[1] for t in tareas if t[0] in plan)

    # Mostramos las tareas no asignadas como advertencia
    if tareas_no_asignadas:
        print(f"  ⚠️  Tareas no asignadas: {', '.join(tareas_no_asignadas)}")
        print()

    return plan, tiempo_total


# ─────────────────────────────────────────────────────────────
# FUNCIÓN PARA MOSTRAR RESULTADOS
# ─────────────────────────────────────────────────────────────

def mostrar_plan(plan: Dict, tiempo_total: int, tareas_originales: List) -> None:
    """
    Muestra el plan de asignación de forma visual y ordenada.

    Formato:
        Tarea A       → CPU   [ 0 - 3 ]  ✅
        Tarea B       → CPU   [ 3 - 4 ]  ✅
        ...

    Donde [inicio - fin] indica el intervalo de tiempo
    en que la tarea está ocupando el recurso.
    """

    print("╔════════════════════════════════════════════════════╗")
    print("║           PLAN DE ASIGNACIÓN (Greedy)             ║")
    print("╚════════════════════════════════════════════════════╝")
    print()

    for nombre_tarea, duracion, _ in sorted(tareas_originales, key=lambda t: t[1]):
        if nombre_tarea in plan:
            recurso, inicio, fin = plan[nombre_tarea]
            icono = "✅"
            print(f"  {nombre_tarea:12s} → {recurso:5s}   [{inicio:2d} - {fin:2d}]  {icono}")
        else:
            print(f"  {nombre_tarea:12s} → {'—':5s}   {'—':>8s}  ❌")

    print()
    print(f"  ⏱️   Tiempo total ejecutado: {tiempo_total}h")
    print(f"  📅  Deadline: {deadline}h")
    print(f"  📊  Tareas asignadas: {len(plan)} / {len(tareas_originales)}")


# ─────────────────────────────────────────────────────────────
# EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔════════════════════════════════════════════════════╗")
    print("║    TASK 1 — Resource Scheduling (Greedy)          ║")
    print("╚════════════════════════════════════════════════════╝")
    print()
    print(f"  📌  Tareas: {len(tareas)}")
    print(f"  🛠️   Recursos: {len(recursos)}")
    print(f"  📅  Deadline: {deadline}h")
    print()

    # ─── Ejecutamos el algoritmo greedy ─────────────────
    plan, tiempo_total = resource_scheduling_greedy(tareas, recursos, deadline)

    # ─── Mostramos el resultado ─────────────────────────
    mostrar_plan(plan, tiempo_total, tareas)

    # ─── Análisis de complejidad ────────────────────────
    print()
    print("╔════════════════════════════════════════════════════╗")
    print("║           ANÁLISIS DE COMPLEJIDAD                  ║")
    print("╚════════════════════════════════════════════════════╝")
    print()
    print("  Algoritmo: Greedy (Shortest Task First)")
    print("  Complejidad: O(N * M)")
    print()
    print("  N = número de tareas")
    print("  M = número de recursos")
    print()
    print("  📘  Explicación:")
    print("       - Ordenar tareas: O(N log N)")
    print("       - Asignar cada tarea: O(N * M)")
    print("       - Total: O(N log N + N * M) ≈ O(N * M)")
    print()
    print("  ⚠️  Limitación: El greedy NO garantiza la solución óptima")
    print("       global. Solo da una solución 'suficientemente buena'")
    print("       de forma rápida.")
    print()
