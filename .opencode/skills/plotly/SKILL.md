---
name: plotly
description: "Visualización interactiva de datos con Plotly Python. Dispara con: gráfico, chart, plot, visualización, dashboard, scatter plot, histograma, heatmap, serie temporal."
---

# Plotly — Visualización Interactiva

## Overview
Librería de gráficos interactivos con 40+ tipos de chart. Ideal para dashboards, EDA y presentaciones.

## Instalación
```bash
uv pip install plotly kaleido
```

## APIs disponibles

### Plotly Express (px) — API de alto nivel
```python
import plotly.express as px
fig = px.scatter(df, x='col1', y='col2', color='category', title='Título')
fig.show()
```

### Graph Objects (go) — Control detallado
```python
import plotly.graph_objects as go
fig = go.Figure(data=[go.Scatter(x=[1,2], y=[3,4])])
fig.update_layout(title='Custom')
```

## Tipos de gráfico principales
- **Básicos**: scatter, line, bar, pie, area, bubble
- **Estadísticos**: histogram, box, violin, distribution
- **Científicos**: heatmap, contour, 3D surface, mesh
- **Financieros**: candlestick, OHLC, time series
- **Mapas**: scatter_map, choropleth
- **Especializados**: sunburst, treemap, sankey, parallel coordinates

## Exportación
```python
fig.write_html('chart.html')        # Interactivo standalone
fig.write_image('chart.png')        # PNG (requiere kaleido)
fig.write_image('chart.pdf')        # PDF
```

## Subplots (múltiples gráficos)
```python
from plotly.subplots import make_subplots
fig = make_subplots(rows=2, cols=2, subplot_titles=('A','B','C','D'))
fig.add_trace(go.Scatter(x=[1,2], y=[3,4]), row=1, col=1)
```

## Referencia
- Documentación: https://plotly.com/python/
- Skill completo: `~/.config/opencode/skill-libraries/data-science/plotly/SKILL.md`
