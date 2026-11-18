# Base técnica de la reescritura

Revisión del código y los artículos disponibles, realizada el 12 de noviembre de 2026. Este documento distingue las capacidades de los pesos originales, las modificaciones que necesitan entrenamiento y las partes de los artículos que no están en el código público guardado aquí.

## Conclusión verificable

La v2 pública **sí está preparada arquitectónicamente para recibir colores locales**. No hay que sustituirla para añadir un pincel: `Generator.to0` recibe cinco canales, y `MangaColorizator.update_hint` construye cuatro de ellos. El contrato es:

```text
entrada = [gris, R_pista * máscara, G_pista * máscara, B_pista * máscara, máscara]
gris: [0, 1]; RGB_pista: [-1, 1]; máscara: [0, 1]
salida RGB: [-1, 1]
```

La máscara es indispensable: un color negro tiene que distinguirse de la ausencia de pista. El extractor estructural recibe solo el canal gris. El decodificador combina sus características con la rama que recibe las pistas. Que exista ese contrato no demuestra por sí solo la precisión con que **un checkpoint concreto** obedece las instrucciones; eso se verifica cambiando las pistas con la imagen fija y midiendo el cambio de salida.

Una **paleta global sin posiciones** no tiene una entrada propia en esta v2. Reordenar colores o dibujarlos en un margen no constituye una interfaz de paleta aprendida. Se necesita un condicionamiento nuevo y entrenamiento. Una lista de colores tampoco determina qué región corresponde a cada color; para especificar «pelo azul, camisa roja» hacen falta posiciones, máscaras o alguna condición semántica adicional.

Evidencia de código original:

- `manga-colorization-v2-master/networks/models.py:191`: `to0` recibe cinco canales.
- `manga-colorization-v2-master/networks/models.py:286`: recorrido de inferencia; `:293` utiliza `sketch[:, 0:1]` en el extractor.
- `manga-colorization-v2-master/colorizator.py:32`: inicialización de pistas vacías; `:34` define `update_hint`; `:52` concatena imagen y pistas.
- `manga-colorization-v2-master/inference.py`: la CLI original nunca llama a `update_hint`; su ausencia en la interfaz no implica ausencia en la red.
- `manga-colorization-v2-master/utils/utils.py`: selección del primer canal RGB y redimensionado propios del código original. Cambiar el preprocesado cambia la distribución de entrada; debe evaluarse, aunque se mejoren validaciones y formatos de imagen.

## Qué aporta cada artículo

| Fuente | Evidencia relevante | Consecuencia para este proyecto |
| --- | --- | --- |
| Golyadkin y Makarov, **Semi-automatic Manga Colorization Using Conditional Adversarial Networks**, 2021 | Sección 3.2, pp. 4–5 del PDF: pistas RGB con máscara; extractor fijo y generador profundo. Sección 4.2, p. 8: 50% de máscaras dispersas, 49% vacías y 1% completas. | La v1 tenía control local real y un esquema reutilizable de simulación de pistas. No es una interfaz nueva que deba inventarse desde cero. |
| Mismo artículo, secciones 4.3–4.5, pp. 9–11 | Advierte del sobreajuste al adaptar con imágenes B/N sin pérdida supervisada y de la propagación de colores a objetos vecinos. Propone conservar pistas de la primera colorización para estabilizar zonas. | Un pincel de pocos píxeles expresa una preferencia; una máscara de región expresa un alcance. No debe prometerse que una pista aislada respete siempre todos los contornos. |
| Golyadkin y Makarov, **Robust Manga Page Colorization via Coloring Latent Space**, 2023 | Sección III, pp. 3–5: un sistema de dos etapas genera un borrador con StyleGAN y lo refina con una red de colorización y pistas. La red recibe gris, pistas de cuatro canales y borrador RGB. | El framework completo del artículo no equivale al `Generator` público de cinco canales. Aquí no están el proyector pSp ni el generador StyleGAN del sistema de borradores. |
| Mismo artículo, sección IV-A y apéndice D, pp. 6–7 y 16 | Síntesis XDoG seguida por traducciones aprendidas hacia B/N real; adaptación adversarial; pseudoetiquetado con varias instantáneas. Entrenamiento alterna la presencia de pistas y borradores. | Convertir todo el color a gris no reproduce manga de tinta y tramas. Una implementación de XDoG es una base útil, pero no una reproducción de su pipeline aprendido completo. |
| Golyadkin, Saraev y Makarov, **Advancing Sequential Manga Colorization for AR Through Data Synthesis**, 2025 | Resumen e introducción, pp. 1–2: benchmark de fotografías de páginas, sombras, perspectiva y síntesis de fotografías mediante difusión. | Aporta futuras degradaciones para fotografías. No es un procedimiento para aprender colores verdaderos a partir de páginas B/N sin etiquetas. Tampoco suministra por sí solo un checkpoint compatible con esta v2. |
| **Neural network methods for manga image processing and enhancement**, resumen de tesis, 2026 | pp. 16–18 distingue mc-v2, borradores y variantes sin borrador; pp. 18–20 describe el paso a pares auténticos. Bibliografía, p. 29, identifica el trabajo WACV 2025 que faltaba. | Confirma que mejorar los datos auténticos y separar paleta de detalle son decisiones centrales. Los experimentos del documento no deben atribuirse automáticamente a los pesos locales. |
| **SensorSCAN: Self-supervised learning and deep clustering for fault diagnosis in chemical processes**, 2023 | Resumen, p. 1: sensores y detección de fallos en procesos químicos. | Comparte autor, pero no proporciona arquitectura ni datos para colorización. Queda fuera de la implementación. |

Los PDF locales se conservan en la raíz con sus nombres originales. La numeración anterior indica páginas del archivo, no necesariamente la numeración editorial impresa. El texto extraído para la revisión está en `tmp/papers/`.

## El artículo que faltaba: pares auténticos

Fuente primaria localizada: [Golyadkin, Plevokas y Makarov, WACV 2025, pp. 5580–5590](https://openaccess.thecvf.com/content/WACV2025/html/Golyadkin_Closing_the_Domain_Gap_in_Manga_Colorization_via_Aligned_Paired_WACV_2025_paper.html), [PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Golyadkin_Closing_the_Domain_Gap_in_Manga_Colorization_via_Aligned_Paired_WACV_2025_paper.pdf).

La sección 3 empareja páginas, detecta texto con MAGI y lo rellena según el fondo; alinea mediante SIFT y RANSAC y filtra coincidencias defectuosas. Para paneles combina detección, SAM y filtrado estructural XDoG+SSIM. La sección 4.1 aclara que la v2 pública es solo un componente del framework anterior. Sus comparaciones favorecen los pares auténticos frente a los sintéticos y distinguen calidad cromática de conservación de estructura. Esto justifica admitir pares B/N–color externos y validar por obras separadas.

Este proyecto no descarga ese dataset. El acceso a un artículo, a un repositorio o a una imagen no acredita una licencia para redistribuir sus imágenes o entrenar un producto. Cada colección incorporada debe conservar procedencia, licencia o autorización y restricciones; también hay que comprobar por separado las condiciones del código y de los pesos. Las copias originales locales no incluyen un archivo `LICENSE` visible que permita asignarles aquí una licencia nueva. No se deriva autorización general de las consideraciones jurídicas descritas por los autores.

## Decisión de arquitectura para la reescritura

Se conserva el generador v2 como base para aprovechar sus pesos y mantener el contrato local de cinco canales. Se incorpora una rama de paleta **DeepSets + FiLM**: cada color pasa por el mismo codificador, una agregación enmascarada produce una representación del conjunto y esta modula características intermedias. La agregación evita asignar significado al orden de los colores y la máscara distingue colores válidos de posiciones de relleno. Es una decisión de esta reescritura; no se atribuye al autor original.

La modulación se inicializa con efecto nulo para que añadir una rama sin entrenar no destruya la salida del modelo base. Esa inicialización **no enseña a usar la paleta**: los parámetros nuevos tienen que aprender con ejemplos. Se entrena con paletas extraídas de las imágenes objetivo, pistas de puntos, trazos y regiones, y ejemplos sin condiciones. El abandono de condiciones evita que el modelo dependa siempre de recibir todas las respuestas.

Para supervisar control local se mide el error cromático dentro de las pistas, además de reconstrucción y coherencia estructural. Una paleta es una orientación global: forzar cada píxel al color RGB más cercano eliminaría sombras y degradados. Las pérdidas de paleta deben equilibrarse con reconstrucción, evaluarse en zonas coloreadas y distinguir preferencias cromáticas de una cuantización obligatoria. Si se solicita un color exacto dentro de una máscara, una composición explícita puede garantizar los píxeles de esa máscara; la propagación aprendida fuera de ella sigue siendo una predicción. Son comportamientos distintos y deben comunicarse así en la interfaz.

## Datos y entrenamiento: qué información existe

| Material disponible | Ejemplo de entrenamiento válido | Límite |
| --- | --- | --- |
| Imagen a color | Objetivo RGB; entrada gris o tinta sintética; paleta y pistas derivadas del objetivo. | La tinta sintetizada no captura todas las tramas y estilos reales. |
| Par auténtico B/N y color de la misma página | Entrada B/N real y objetivo RGB alineados; registrar transformación y revisar el recorte común. | Coincidir por nombre o tamaño no demuestra alineamiento; páginas redibujadas pueden ser inutilizables. |
| Solo B/N | Generar una propuesta con el modelo anterior, conservarla como pseudoetiqueta y revisarla. | Los colores son hipótesis, no verdad de referencia. Repetir los errores del profesor puede consolidarlos. |
| B/N con anotaciones de color | Supervisión parcial de las áreas anotadas, o completar y revisar el objetivo. | Las zonas sin etiqueta no adquieren colores verdaderos por el mero hecho de entrenar. |

La preparación de datos debe poder construirse de nuevo desde un manifiesto, sin modificar los originales. Debe registrar origen, pareja, tipo de objetivo —auténtico, sintético o pseudoetiqueta— y grupo de separación. Mantener páginas de una misma obra/capítulo fuera de particiones distintas reduce la fuga de información; las pruebas de generalización más exigentes separan obras completas. Los pseudoobjetivos se revisan y se ponderan menos que los auténticos.

Empezar desde el checkpoint v2 y ajustar con supervisión explícita es una base más directa que reconstruir StyleGAN, pSp, CUT y todos los entrenamientos del artículo de 2023. El entrenamiento adversarial y perceptual puede añadirse o ajustarse después de verificar que el modelo aprende control, sin presentar un ajuste corto como reproducción de todos los resultados publicados.

## Compatibilidad que se promete

- **Pesos v2 originales → nueva base:** carga estricta en el generador original. Los parámetros de paleta son adicionales y quedan identificados como nuevos. Una carga parcial silenciosa no acredita compatibilidad.
- **Checkpoint nuevo → aplicación nueva:** formato explícito con versión de arquitectura, pesos del generador y de paleta, configuración de preprocesado y estado de entrenamiento necesario para reanudar.
- **Exportación a v2 original:** puede extraerse el `state_dict` del generador para su interfaz de cinco canales. La aplicación original no conoce DeepSets/FiLM; la exportación **pierde la capacidad de paleta global** y puede tener otra calidad por haberse entrenado junto con ella. Debe evaluarse por separado.
- **Pesos v1 → v2:** no son intercambiables. La v1 utiliza también un campo de distancia y una arquitectura distinta; sus ejemplos de datos y su lógica de pistas sirven de referencia, no de conversor de pesos.

## Verificación antes de afirmar una mejora

El software puede quedar implementado y probado antes de disponer de un modelo nuevo de calidad. Deben conservarse separados esos dos hitos. Un paso de entrenamiento que termina sin errores prueba el flujo de gradientes y el formato de checkpoint, no la calidad artística.

La evaluación mínima compara el checkpoint original con el nuevo en imágenes reservadas: salida sin pistas, pistas de colores opuestos en la misma posición, paleta cambiada con imagen fija, orden de paleta permutado, pistas negras y blancas, y regiones pequeñas y grandes. Medir error dentro de máscara, cambios fuera de ella y preservación de tinta. Guardar ejemplos y parámetros de ejecución permite comprobar tanto el control conseguido como los casos donde el modelo todavía falla.
