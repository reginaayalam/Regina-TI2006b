# Databricks notebook source
# MAGIC %md
# MAGIC # Basic Statistics – MLOps in Databricks (TI2006B)
# MAGIC **Alumna:** Regina Ayala Merla · A00841585
# MAGIC
# MAGIC **Pregunta:** ¿Cómo varían los montos de las transacciones entre canales (Online vs. Store)?
# MAGIC
# MAGIC **Fuente de datos:** dataset sintético generado en este notebook con `numpy.random.default_rng(42)` (no contiene datos privados ni sensibles).
# MAGIC
# MAGIC **Unidades:** `amount` en unidades sintéticas; `channel` es categórica (Online / Store).
# MAGIC
# MAGIC **Política de valores faltantes:** se excluyen las filas con `amount` o `channel` nulos (`dropna`). No se imputa ningún valor.

# COMMAND ----------

# MAGIC %pip install plotly

# COMMAND ----------

import numpy as np
import pandas as pd
import plotly
import plotly.express as px

rng = np.random.default_rng(42)
df = pd.DataFrame({
    "channel": ["Online"] * 100 + ["Store"] * 100,
    "amount": np.round(rng.lognormal(3.5, 0.6, 200), 2),
})
df.loc[0, "amount"] = np.nan

print("Versions:", np.__version__, pd.__version__, plotly.__version__)
print("Rows:", len(df))
print("Missing values:", df[["amount", "channel"]].isna().sum())

# COMMAND ----------

clean = df.dropna(subset=["amount", "channel"]).copy()
if len(clean) < 2 or not np.isfinite(clean["amount"]).all():
    raise ValueError("Need at least two finite amounts")

print("Analyzed rows:", len(clean))
print(clean["amount"].describe().round(2))
print("Median:", round(clean["amount"].median(), 2))

grouped = clean.groupby("channel")["amount"].agg(["count", "mean", "median", "std"]).round(2)
print(grouped)

# Reconciliación de conteos: los grupos deben sumar las filas analizadas
assert grouped["count"].sum() == len(clean), "Grouped counts do not match analyzed rows"
print("Grouped counts sum:", grouped["count"].sum(), "== analyzed rows:", len(clean))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resultados de las estadísticas
# MAGIC - **Filas totales:** 200 · **Faltantes en `amount`:** 1 · **Filas analizadas:** 199
# MAGIC - **Media:** 37.61 · **Mediana:** 32.06 · **Desviación estándar (muestral):** 23.12
# MAGIC - **Mínimo:** 9.21 · **Q1:** 22.31 · **Q3:** 45.72 · **Máximo:** 190.25
# MAGIC - **Por canal:** Online 99 filas (media 35.65, mediana 32.78) · Store 100 filas (media 39.55, mediana 30.64). 99 + 100 = 199 ✔
# MAGIC
# MAGIC **Exclusión:** se eliminó 1 fila porque `amount` estaba vacío (se forzó a NaN en la fila 0, canal Online). Al ser 1 de 200 (0.5%), el efecto en los resultados es mínimo, pero deja al grupo Online con una fila menos (99 vs. 100).

# COMMAND ----------

histogram = px.histogram(
    clean, x="amount", nbins=20, title="Transaction amount distribution",
    labels={"amount": "Amount (synthetic units)"},
)
box = px.box(
    clean, x="channel", y="amount", title="Amounts by channel",
    labels={"amount": "Amount (synthetic units)", "channel": "Channel"},
)
histogram.show()
box.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interpretación y conclusiones
# MAGIC **Histograma:** la distribución está sesgada a la derecha (cola larga hacia montos altos). La mayoría de las transacciones se concentra entre ~20 y ~45 unidades, con unos cuantos valores altos que llegan a 190.
# MAGIC
# MAGIC **Observación con estadísticas:** la media (37.61) es mayor que la mediana (32.06), lo que confirma el sesgo a la derecha: unos pocos montos altos jalan la media hacia arriba, así que la mediana representa mejor una transacción "típica".
# MAGIC
# MAGIC **Observación con gráfica:** en el box plot, las medianas de Online y Store son muy parecidas (32.78 vs. 30.64), pero Store muestra más valores atípicos altos y mayor dispersión, lo que explica su media más alta (39.55).
# MAGIC
# MAGIC **Tamaños de grupo:** ligeramente desiguales (Online 99, Store 100) por la fila excluida.
# MAGIC
# MAGIC **Limitación:** los datos son sintéticos y ambos canales se generan con la misma distribución (lognormal 3.5, 0.6), así que las diferencias observadas son ruido aleatorio y no un efecto real del canal. Además, con 100 filas por grupo unos pocos atípicos pueden mover mucho la media.
# MAGIC
# MAGIC **Conclusión:** no hay evidencia de que el canal cambie el monto típico de las transacciones; la diferencia en medias se debe a valores atípicos.
