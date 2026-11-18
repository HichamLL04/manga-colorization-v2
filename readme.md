# MangaColor: control de color sobre v2

Reescritura en Python con editor local, pistas por zonas, paleta global y un flujo de datos/entrenamiento propio. Las carpetas originales v1/v2 se conservan como referencia.

La [especificación completa del diseño de software](docs/SSD.md) documenta la arquitectura actual y objetivo, los contratos, los parámetros y los criterios de aceptación.

## Estado real

- **Disponible con `v2generator.zip`:** colorización automática, pistas RGB mediante puntos, pincel y máscara de región; paleta aproximada mediante una segunda pasada con pistas automáticas.
- **Modelo extendido implementado:** v2 + adaptador global de paleta DeepSets/FiLM, inicializado sin modificar la salida del checkpoint original. El adaptador aprende de paletas y objetivos durante el entrenamiento.
- **Entrenamiento disponible:** datos desde color, parejas B/N-color alineadas o pseudoetiquetas de B/N; pérdidas de reconstrucción, salida auxiliar, cumplimiento de pistas y preferencia cromática de paleta. No es una reproducción de todo el GAN del artículo.
- **No hay un nuevo checkpoint de calidad artística validada.** Las pruebas cortas verifican gradientes, ejecución, control y compatibilidad; no acreditan una mejora visual general. Los ejemplos incluidos en v2 no sustituyen un conjunto de entrenamiento y evaluación.

La v2 ya recibe **5 canales**: gris `[0,1]`, RGB de pistas normalizado `[-1,1]` multiplicado por máscara y máscara `[0,1]`. La CLI original no exponía estas pistas. Los pesos v1 de 6 canales y los TorchScript `colorizer.zip` se rechazan expresamente.

## Instalación y arranque en Windows

Desde `E:\GitHub\manga-colorization`, el entorno `.venv` ya está preparado con PyTorch CPU para las verificaciones de esta reescritura:

```powershell
.\.venv\Scripts\python.exe -m mangacolor inspect v2generator.zip
.\.venv\Scripts\python.exe -m mangacolor edit --checkpoint v2generator.zip
```

Abre [el editor local](http://127.0.0.1:7860). Carga una página, elige el color y pinta. El relleno encuentra regiones cerradas a partir de la tinta; un contorno abierto puede desbordarse, y se puede deshacer. Las pistas se descargan como PNG RGBA y conservan las coordenadas de la imagen original. El resultado conserva su tamaño original aunque la red calcule a menor resolución.

Para instalar desde cero:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
```

La RTX 4050 detectada tiene 6 GB. Para entrenar con ella, instala una distribución de PyTorch con CUDA siguiendo el [instalador oficial](https://pytorch.org/get-started/locally/) y verifica `torch.cuda.is_available()`. El entorno de verificación actual es CPU; `--device cuda` requiere esa instalación. Empieza con tamaño 256, lote 1 y `--scope adapter` si necesitas reducir memoria. No se ha medido aquí el consumo de entrenamiento CUDA.

## Colorización y control

```powershell
.\.venv\Scripts\python.exe -m mangacolor colorize pagina.png resultado.png --hints pistas.png --lock-hints
.\.venv\Scripts\python.exe -m mangacolor colorize pagina.png resultado_paleta.png --palette '#d89b75' '#3267a8' '#6f4050'
```

`pistas.png` debe tener las dimensiones del original. Su alfa indica dónde hay color; el negro también es una pista válida. Sin `--lock-hints`, la salida es la predicción del modelo. Con esa opción se compone el color exactamente en los píxeles de alfa 255; un alfa intermedio mezcla colores. El bloqueo se aplica después de `--preserve-ink`, por lo que tiene prioridad incluso encima de una línea negra. Pinta dentro de los contornos si quieres conservarlos.

La paleta predeterminada `--palette-mode hints` **no es un adaptador ya entrenado**: genera una propuesta automática, escoge muestras cromáticas y solicita colores próximos de la paleta en una segunda pasada. Es una preferencia aproximada; no garantiza usar exclusivamente esos colores ni asignarlos a objetos concretos. El informe JSON registra cuántas pistas pudo colocar. Las pistas manuales tienen prioridad en sus píxeles.

Para la paleta aprendida, usa un checkpoint de entrenamiento:

```powershell
.\.venv\Scripts\python.exe -m mangacolor colorize pagina.png resultado_modelo.png --checkpoint runs\experimento\last.pt --palette '#d89b75' '#3267a8' --palette-mode learned
```

El programa comprueba que el adaptador haya recibido actualizaciones; no confunde un módulo recién añadido con pesos entrenados. Una paleta es un conjunto de colores, sin etiquetas semánticas. Para «pelo azul, camiseta roja», usa las pistas espaciales junto con la paleta.

## Crear los datos

```powershell
.\.venv\Scripts\python.exe -m mangacolor prepare D:\datos\color data\manga
```

Crea `manifest.jsonl`, objetivos y variantes de gris/tinta sintética sin modificar los originales. Estas variantes son una base de entrenamiento, no una reproducción del pipeline XDoG/CUT de los papers. Se generan paletas, puntos, trazos y máscaras de regiones durante el entrenamiento; las recoloraciones HSV del objetivo mantienen coherentes sus pistas y paletas.

Las subcarpetas superiores pueden representar obras. Si hay varias, se separan obras completas entre entrenamiento y validación. Con una sola obra se separan imágenes por hash; no mide generalización a otras obras. Las variantes y los duplicados exactos permanecen en la misma partición. El manifiesto registra procedencia, tipo de supervisión y hashes.

Para pares auténticos:

```powershell
.\.venv\Scripts\python.exe -m mangacolor prepare D:\datos\color data\pareados --bw-dir D:\datos\blanco_negro
```

Se empareja por ruta relativa y nombre sin extensión; se rechazan tamaños diferentes. **Mismo tamaño no demuestra alineamiento:** revisa que la tinta y el color correspondan píxel a píxel. El programa no inventa un registro SIFT/RANSAC ni alinea redibujados automáticamente.

Si solo tienes B/N:

```powershell
.\.venv\Scripts\python.exe -m mangacolor prepare D:\datos\bw data\sin_etiquetas
.\.venv\Scripts\python.exe -m mangacolor pseudo-label data\sin_etiquetas\manifest.jsonl data\pseudo --checkpoint v2generator.zip
```

Las imágenes monocromas se registran sin objetivo. `pseudo-label` crea propuestas del profesor marcadas `pseudo`, con peso 0,25 y hash del checkpoint. No convierte colores hipotéticos en etiquetas reales; revisa/corrige las propuestas antes de una sesión extensa. Las obras reservadas para validación no se incorporan como pseudoentrenamiento. Para obtener una validación fiable necesitas color auténtico reservado, no la salida del profesor. Los datos B/N sin objetivo no entran en la pérdida supervisada.

## Entrenar y exportar

```powershell
.\.venv\Scripts\python.exe -m mangacolor train data\manga\manifest.jsonl runs\experimento --checkpoint v2generator.zip --size 256 --batch-size 1 --epochs 10 --device cpu
```

`--scope decoder` congela el extractor semántico y ajusta el resto del generador junto con el adaptador. `--scope adapter` ajusta solo la paleta y conserva todos los pesos v2, por lo que no mejora por sí solo la propagación de pistas locales. En CUDA se usa precisión mixta y escalado de gradientes. No se descarga VGG ni ningún modelo externo.

`last.pt` guarda generador, adaptador, optimizador y metadatos. `metrics.jsonl` guarda pérdidas y validación; `config.json` la configuración. `--max-steps 3` limita una prueba mecánica. Los checkpoints se marcan `experimental` y `quality_validated=false`; haber dado un paso de optimización no implica calidad. No se sobrescribe un entrenamiento previo sin `--resume`.

```powershell
.\.venv\Scripts\python.exe -m mangacolor train data\manga\manifest.jsonl runs\experimento --resume runs\experimento\last.pt --scope decoder --epochs 5
.\.venv\Scripts\python.exe -m mangacolor export-v2 runs\experimento\last.pt runs\generator_v2.zip
```

La reanudación recupera pesos, optimizador y número de época, y comienza una época nueva; no reproduce exactamente la secuencia de un lote interrumpido ni el estado del escalador AMP. Usa el mismo dataset, tamaño, alcance y parámetros. La exportación tiene exactamente los nombres y dimensiones del `Generator` original: admite pistas locales, pero **no incluye el adaptador de paleta global**. Conserva `last.pt` para esa capacidad y evalúa la calidad del export por separado.

## Reconstrucción a alta resolución y desacoplo de escalas (Hito B)

```powershell
# Predecir a escala canónica (576) y exportar a alta resolución conservando el trazo original:
.\.venv\Scripts\python.exe -m mangacolor colorize pagina.png resultado_hd.png --export-size 2250 --save-color-field campo.npz

# Proteger texto original con error cero (máscara binaria):
.\.venv\Scripts\python.exe -m mangacolor colorize pagina.png resultado_texto.png --text-mask mascara_texto.png

# Reutilizar el campo canónico predicho para exportar a otro tamaño SIN volver a invocar la red:
.\.venv\Scripts\python.exe -m mangacolor reconstruct pagina.png campo.npz resultado_otra_resolucion.png --export-size 1125
```

El desacoplo (Sección 11 del SSD) calcula el campo de color canónico (`ColorField`) en espacio CIELAB a escala de trabajo fija (576 píxeles). Al exportar a cualquier resolución:
1. Las componentes cromáticas $a^*, b^*$ se amplían mediante un **Guided Filter** guiado por la estructura de luminancia del dibujo original.
2. La **descomposición de luminosidad** ($L$) preserva la frecuencia y nitidez de la tinta original combinándola con la iluminación suave calculada por la red.
3. Se aplica **gamut mapping** hacia el eje neutro para mantener los colores dentro de sRGB sin artefactos de recorte ni distorsión tonal.
4. Las máscaras de texto y tinta protegidas prevalecen con **error RGB 0.0**.

## Comprobaciones

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\verify_pipeline.py
```

La segunda orden entrena tres pasos sobre figuras geométricas generadas y comprueba que el adaptador cambie la salida y que la exportación cargue estrictamente en v2. Es una prueba del circuito, no un benchmark artístico. Escribe en `output/pipeline_check` y requiere un destino nuevo.

La prueba del checkpoint real está en `output/verification/model_verification.json`, con comparación rojo/azul en `local_hints_comparison.png`. La salida sin paleta del generador extendido coincide exactamente con la implementación original usando el mismo tensor de entrada. El preprocesado nuevo usa luminancia real; el antiguo tomaba el primer canal, así que para originales RGB de color no son recorridos idénticos.

Consulta [la revisión técnica](docs/RESEARCH.md) para la relación con los papers y el [artículo de WACV 2025 sobre datos auténticos alineados](https://openaccess.thecvf.com/content/WACV2025/html/Golyadkin_Closing_the_Domain_Gap_in_Manga_Colorization_via_Aligned_Paired_WACV_2025_paper.html). El código/pesos originales y las imágenes requieren revisar sus condiciones de uso; esta reescritura no les asigna una licencia nueva.
