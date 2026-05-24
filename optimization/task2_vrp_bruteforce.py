"""
╔══════════════════════════════════════════════════════════════╗
║  TASK 2 — Vehicle Routing Problem (Brute Force)             ║
║  Optimización de rutas de vehículos por fuerza bruta        ║
║                                                             ║
║  Algoritmo: Brute Force (enumeración exhaustiva)            ║
║  Concepto: probar TODAS las opciones posibles y             ║
║            quedarse con la mejor                            ║
╚══════════════════════════════════════════════════════════════╝

PROBLEMA:
  Tenemos:
    - Un depot (punto de partida y llegada de cada vehículo)
    - N clientes (cada uno en una coordenada del plano)
    - K vehículos (cada uno puede visitar varios clientes)

  Objetivo: encontrar la ruta de menor distancia total que:
    - Cada vehículo sale del depot y vuelve al depot
    - Todos los clientes son visitados exactamente una vez
    - Cada vehículo tiene un límite de clientes que puede visitar

ALGORITMO BRUTE FORCE:
  1. Generar TODAS las permutaciones posibles de los clientes
     → itertools.permutations(lista_clientes)
  2. Para cada permutación:
     a. Dividir la permutación en segmentos según la capacidad
        de cada vehículo
     b. Para cada segmento (ruta de un vehículo):
        - Calcular distancia: depot → c1 → c2 → ... → depot
     c. Sumar todas las rutas para obtener distancia total
  3. Comparar con la mejor solución encontrada
  4. Retornar la que tiene menor distancia

COMPLEJIDAD: O(N! / (K! * (C!)^K)) ≈ O(N!) en el peor caso
  ⚠️ Solo factible para N ≤ 10 clientes

DIFERENCIA CON GREEDY:
  - Greedy: toma decisiones locales rápidas, solución buena pero no óptima
  - Brute Force: prueba todo, encuentra el óptimo global pero es MUY lento
"""

import itertools
import math
import time
from typing import List, Tuple


# ─────────────────────────────────────────────────────────────
# DATOS DE ENTRADA
# ─────────────────────────────────────────────────────────────

# Coordenadas de los puntos en el plano (x, y).
# El depot es donde todos los vehículos comienzan y terminan.
depot: Tuple[float, float] = (0, 0)  # Coordenada del depot

# Lista de clientes. Cada cliente es una coordenada (x, y).
clientes: List[Tuple[float, float]] = [
    (2, 3),   # Cliente 1
    (5, 1),   # Cliente 2
    (1, 5),   # Cliente 3
    (6, 4),   # Cliente 4
    (3, 2),   # Cliente 5
]

# Número de vehículos disponibles
num_vehiculos: int = 2

# Capacidad de cada vehículo (máximo de clientes por ruta)
capacidad_vehiculo: int = 3

# ─── Nota: Con 5 clientes, 5! = 120 permutaciones (factible) ───
# 5! = 120 permutaciones → se resuelve al instante
# 10! = 3,628,800 → ya empieza a tardar segundos
# 12! = 479M → minutos/horas, inviable en la práctica


# ─────────────────────────────────────────────────────────────
# FUNCIÓN AUXILIAR: DISTANCIA EUCLIDIANA
# ─────────────────────────────────────────────────────────────

def distancia(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calcula la distancia euclidiana entre dos puntos.

    Fórmula: d = √((x₂ - x₁)² + (y₂ - y₁)²)

    Por qué usamos esta distancia:
        - Es la distancia en línea recta (la más simple)
        - Funciona bien para problemas geométricos
        - Se puede reemplazar por distancia Manhattan o real
          si tenemos datos de calles/rutas

    Ejemplo:
        distancia((0, 0), (3, 4)) = √(9 + 16) = √25 = 5
    """
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


# ─────────────────────────────────────────────────────────────
# FUNCIÓN AUXILIAR: CALCULAR DISTANCIA DE UNA RUTA
# ─────────────────────────────────────────────────────────────

def distancia_ruta(ruta: List[int], todos_clientes: List[Tuple[float, float]]) -> float:
    """
    Calcula la distancia total de una ruta que:
      - Sale del depot
      - Visita los clientes en orden
      - Vuelve al depot

    ruta: lista de índices de clientes (ej: [0, 2, 1])
    todos_clientes: lista con las coordenadas de todos los clientes

    Ejemplo con clientes = [(2,3), (5,1), (1,5)] y ruta = [0, 2, 1]:
        depot → cliente0 → cliente2 → cliente1 → depot
        (0,0) → (2,3) → (1,5) → (5,1) → (0,0)

    La función suma cada tramo de este recorrido.
    """
    total = 0.0

    # 1. Distancia del depot al PRIMER cliente de la ruta
    total += distancia(depot, todos_clientes[ruta[0]])

    # 2. Distancias entre clientes consecutivos en la ruta
    for i in range(1, len(ruta)):
        total += distancia(todos_clientes[ruta[i - 1]], todos_clientes[ruta[i]])

    # 3. Distancia del ÚLTIMO cliente de vuelta al depot
    total += distancia(todos_clientes[ruta[-1]], depot)

    return total


# ─────────────────────────────────────────────────────────────
# IMPLEMENTACIÓN DEL ALGORITMO BRUTE FORCE
# ─────────────────────────────────────────────────────────────

def vrp_bruteforce(
    clientes: List[Tuple[float, float]],
    num_vehiculos: int,
    capacidad_vehiculo: int
) -> Tuple[List[List[int]], float, int]:
    """
    Resuelve el VRP por fuerza bruta (enumeración exhaustiva).

    Parámetros:
        clientes: lista de coordenadas [(x1,y1), (x2,y2), ...]
        num_vehiculos: cantidad de vehículos disponibles
        capacidad_vehiculo: máximo de clientes por vehículo

    Retorna:
        (mejores_rutas, mejor_distancia, total_permutaciones)
            mejores_rutas: lista de listas, cada sublista son los
                          índices de clientes que visita ese vehículo
            mejor_distancia: distancia total mínima encontrada
            total_permutaciones: cuántas permutaciones se evaluaron

    Cómo funciona:
        1. Generamos TODAS las permutaciones posibles de los clientes
           → Para N clientes hay N! permutaciones
        2. Para cada permutación:
           a. Cortamos la permutación en segmentos según la capacidad
              Ej: clientes [0,1,2,3,4], capacidad 3:
                Vehículo 1: [0,1,2] (3 clientes)
                Vehículo 2: [3,4]   (2 clientes, el resto)
           b. Calculamos la distancia de cada ruta individual
           c. Sumamos todo
           d. Si es menor que la mejor encontrada → guardamos
        3. Retornamos la mejor solución

    Por qué es O(N!):
        - N! permutaciones (crece factorialmente)
        - Para cada una dividimos y calculamos distancias O(N)
        - Total: O(N! * N) ≈ O(N!)

    Cuándo usar:
        - Solo cuando N ≤ 10 (N! crece demasiado rápido)
        - Cuando NECESITAMOS la solución exacta (óptimo global)
        - Para validación de otros algoritmos (benchmark)
    """
    n = len(clientes)
    indices = list(range(n))  # [0, 1, 2, ..., n-1]

    mejor_distancia = float('inf')
    mejores_rutas = []
    total_permutaciones = 0

    # 1. Generamos TODAS las permutaciones posibles
    inicio = time.time()

    for permutacion in itertools.permutations(indices):
        total_permutaciones += 1

        # 2. Dividimos la permutación entre los vehículos
        #    Cada vehículo recibe hasta capacidad_vehiculo clientes
        rutas = []
        idx = 0
        for v in range(num_vehiculos):
            # Desde idx hasta idx + capacidad_vehiculo
            fin = min(idx + capacidad_vehiculo, n)
            ruta = list(permutacion[idx:fin])
            if ruta:  # Solo agregamos si la ruta no está vacía
                rutas.append(ruta)
            idx = fin

        # 3. Calculamos la distancia total de esta solución
        distancia_total = 0.0
        for ruta in rutas:
            distancia_total += distancia_ruta(ruta, clientes)

        # 4. Si es la mejor hasta ahora, la guardamos
        if distancia_total < mejor_distancia:
            mejor_distancia = distancia_total
            mejores_rutas = rutas

    fin = time.time()

    print(f"  ⏱️   Tiempo de ejecución: {fin - inicio:.4f} segundos")

    return mejores_rutas, mejor_distancia, total_permutaciones


# ─────────────────────────────────────────────────────────────
# FUNCIÓN PARA MOSTRAR RESULTADOS
# ─────────────────────────────────────────────────────────────

def mostrar_rutas(
    rutas: List[List[int]],
    distancia_total: float,
    total_permutaciones: int
) -> None:
    """
    Muestra las rutas encontradas de forma visual.

    Formato:
        🚛 Vehículo 1: Depot → Cliente 3 → Cliente 5 → Depot  (distancia: 12.5)

    Donde cada "→" indica un tramo entre puntos consecutivos.
    """

    print()
    print("╔════════════════════════════════════════════════════╗")
    print("║         RUTAS ÓPTIMAS ENCONTRADAS (Brute Force)   ║")
    print("╚════════════════════════════════════════════════════╝")
    print()

    for i, ruta in enumerate(rutas):
        # Construimos la representación de la ruta
        partes_ruta = ["Depot"]
        for idx in ruta:
            partes_ruta.append(f"Cliente {idx + 1}{clientes[idx]}")
        partes_ruta.append("Depot")

        ruta_str = " → ".join(partes_ruta)
        dist_ruta = distancia_ruta(ruta, clientes)

        print(f"  🚛 Vehículo {i + 1}:")
        print(f"     {ruta_str}")
        print(f"     Distancia: {dist_ruta:.2f}")
        print()

    print(f"  📏  Distancia TOTAL: {distancia_total:.2f}")
    print()
    print(f"  🔄  Permutaciones evaluadas: {total_permutaciones:,}")
    print()


# ─────────────────────────────────────────────────────────────
# EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("╔════════════════════════════════════════════════════╗")
    print("║   TASK 2 — Vehicle Routing Problem (Brute Force)  ║")
    print("╚════════════════════════════════════════════════════╝")
    print()
    print(f"  📍  Depot: {depot}")
    print(f"  👥  Clientes: {len(clientes)}")
    for i, c in enumerate(clientes):
        print(f"       Cliente {i + 1}: {c}")
    print(f"  🚛  Vehículos: {num_vehiculos}")
    print(f"  📦  Capacidad por vehículo: {capacidad_vehiculo} clientes")
    print()

    # ─── Verificamos que sea factible ─────────────────────
    # Cada vehículo tiene C clientes como máximo
    # Con K vehículos podemos visitar K * C clientes como máximo
    max_clientes = num_vehiculos * capacidad_vehiculo
    if len(clientes) > max_clientes:
        print(f"  ❌ Error: hay {len(clientes)} clientes pero solo podemos "
              f"visitar {max_clientes} (K={num_vehiculos} * C={capacidad_vehiculo})")
        exit(1)

    # ─── Advertencia por cantidad de permutaciones ────────
    n = len(clientes)
    print(f"  ⚠️   N! = {n}! = {math.factorial(n):,} permutaciones")
    if n > 8:
        print("      ⚠️  Esto puede tardar varios segundos/minutos!")
    print()

    # ─── Ejecutamos la fuerza bruta ───────────────────────
    rutas, distancia_total, total_perm = vrp_bruteforce(
        clientes, num_vehiculos, capacidad_vehiculo
    )

    # ─── Mostramos el resultado ───────────────────────────
    mostrar_rutas(rutas, distancia_total, total_perm)

    # ─── Análisis de complejidad ──────────────────────────
    print("╔════════════════════════════════════════════════════╗")
    print("║           ANÁLISIS DE COMPLEJIDAD                  ║")
    print("╚════════════════════════════════════════════════════╝")
    print()
    print("  Algoritmo: Brute Force (enumeración exhaustiva)")
    print("  Complejidad: O(N! / (K! * (C!)^K)) ≈ O(N!)")
    print()
    print(f"  Para este caso: {n}! = {math.factorial(n):,} permutaciones")
    print()
    print("  📘  Explicación:")
    print("       - La fuerza bruta genera TODAS las permutaciones")
    print("       - Cada permutación representa un orden de visita")
    print("       - Divide cada permutación en K segmentos (vehículos)")
    print("       - Calcula la distancia de cada segmento")
    print("       - Guarda la de menor distancia total")
    print()
    print("  ⚠️  Crecimiento factorial:")
    print("       5!  = 120 permutaciones     → instantáneo")
    print("       8!  = 40,320 permutaciones  → 1-2 segundos")
    print("       10! = 3,628,800 permutaciones → varios segundos")
    print("       12! = 479M permutaciones     → minutos/horas")
    print("       15! = 1.3B permutaciones     → inviable!")
    print()
    print("  💡  Para problemas grandes se usan:")
    print("       - Algoritmos genéticos")
    print("       - Búsqueda tabú")
    print("       - Recocido simulado (Simulated Annealing)")
    print("       - Colonia de hormigas (Ant Colony)")
    print()
