---
name: polars
description: "DataFrame rápido en memoria con Polars. Alternativa a pandas con evaluación lazy y ejecución paralela. Dispara con: DataFrame, transformar datos, ETL, polars, lazy, leer CSV, group by, join."
---

# Polars — DataFrames Rápidos

## Overview
Librería DataFrame ultrarrápida construida sobre Apache Arrow. Evaluación lazy, ejecución paralela, API basada en expresiones.

## Instalación
```bash
uv pip install polars
```

## Conceptos clave

### Expresiones (building blocks)
```python
import polars as pl
df = pl.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30], "city": ["NY", "LA"]})
df.select("name", "age")
df.filter(pl.col("age") > 25)
df.with_columns(age_plus_10=pl.col("age") + 10)
```

### Lazy vs Eager
```python
# Eager: ejecuta inmediato
df = pl.read_csv("file.csv")

# Lazy: construye plan optimizado
lf = pl.scan_csv("file.csv")
result = lf.filter(pl.col("age") > 25).select("name").collect()
```

## Operaciones comunes

### Group By y agregaciones
```python
df.group_by("city").agg(
    pl.col("age").mean().alias("avg_age"),
    pl.len().alias("count")
)
```

### Joins
```python
df1.join(df2, on="id", how="inner")
df1.join(df2, left_on="user_id", right_on="id", how="left")
```

### Window functions
```python
df.with_columns(avg_age_by_city=pl.col("age").mean().over("city"))
```

### Pivot / Unpivot
```python
df.pivot(values="sales", index="date", columns="product")
df.unpivot(index="id", on=["col1", "col2"])
```

## Migración desde pandas
| Operación | pandas | polars |
|-----------|--------|--------|
| Select columna | `df["col"]` | `df.select("col")` |
| Filtrar | `df[df["col"] > 10]` | `df.filter(pl.col("col") > 10)` |
| Agregar col | `df.assign(x=...)` | `df.with_columns(x=...)` |
| Group by | `df.groupby("col")` | `df.group_by("col")` |

## Buenas prácticas
- Usar lazy (`scan_csv`) para datasets grandes
- Seleccionar columnas necesarias temprano
- Preferir API de expresiones sobre `.map_elements()`
- Usar `streaming=True` para datos muy grandes
- Usar tipos Categorical para strings de baja cardinalidad

## Formatos soportados
CSV, Parquet, JSON, Excel, bases de datos, S3, Azure, GCS, BigQuery

## Referencia
- Documentación: https://pola.rs/
- Skill completo: `~/.config/opencode/skill-libraries/data-science/polars/SKILL.md`
