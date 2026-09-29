# TP - Clasificación Multiclase de Texto con Regresión Softmax

Trabajo práctico de la materia **Redes Neuronales en Bioingeniería** (ITBA). Implementa un clasificador
de texto multiclase (Regresión Softmax en PyTorch) sobre el dataset **20 Newsgroups**
(~18.000 posts de foros de Usenet en 20 categorías), con un pipeline de preprocesamiento
configurable, vectorización TF-IDF, búsqueda de hiperparámetros con Optuna, y tracking
de experimentos con MLflow y TensorBoard.

Consigna base: [TP-Regresion-Softmax-TNG](https://github.com/cselmo/TP-Regresion-Softmax-TNG)

## Cómo correrlo

1. Abrir `TP_Regresion_Softmax_TNG.ipynb` en [Google Colab](https://colab.research.google.com/).
2. Ejecutar las celdas en orden desde arriba (`Entorno de ejecución > Ejecutar todas`).
   La primera celda instala las dependencias que no vienen por defecto en Colab
   (`mlflow`, `optuna`).
3. La celda de precálculo de preprocesamiento (sección 6) y las dos búsquedas de Optuna
   (secciones 6 y 10, 15 trials cada una) son los pasos que más tardan.

## Estructura del notebook

1. Datos: carga de 20 Newsgroups
2. Preprocesamiento configurable (stopwords, stemming/lematización)
3. Vectorización TF-IDF
4. Modelo: Regresión Softmax (PyTorch)
5. Entrenamiento
6. Búsqueda de hiperparámetros con Optuna
7. Entrenamiento final
8. Evaluación final
9. Preguntas de reflexión
10. Análisis extendido: resultados de la búsqueda, dinámica de entrenamientos y
    tiempos de convergencia

## Resultados principales

- Mejor combinación encontrada por Optuna: sacando stopwords, stemming, unigramas, `max_features=10000`,
  `min_df` entre 2 y 5.
- Accuracy en test (modelo final): **~0.6569**
- Las categorías con más confusión son las de religión (`talk.religion.misc`,
  `alt.atheism`), política (`talk.politics.*`) y los subgrupos de `comp.`,
  detallado en la sección 9 (pregunta 9) y 10 del notebook.
