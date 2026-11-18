# Especificación del diseño de software de MangaColor

Documento SSD de la reescritura de manga colorization v2

Versión 1.0 | Fecha 16 de noviembre de 2026 | Idioma español

## 1 Propósito y autoridad del documento

Este SSD especifica el software de colorización de manga con control de color local, paletas globales, preparación de datos y entrenamiento compatible con el generador público v2. También define la ampliación necesaria para conservar el detalle original al exportar a alta resolución y mantener colores de personajes entre viñetas.

El resultado buscado es una aplicación local donde el usuario pueda decidir colores, corregir regiones y reutilizar decisiones cromáticas sin que el tamaño de exportación provoque una nueva interpretación de la página. El modelo debe poder entrenarse con datos propios y exportar su generador al formato nativo v2, conservando explícitamente los límites de esa compatibilidad.

El documento describe dos estados separados. **Implementado** identifica comportamiento presente en el código de la versión 0.1.0. **Objetivo** identifica requisitos y contratos que deben desarrollarse para completar el producto. La publicación de este SSD no convierte los objetivos en funciones disponibles ni acredita la calidad de un modelo sin evaluar.

Las palabras **debe**, **no debe** y **se rechazará** expresan requisitos verificables. **Puede** expresa una opción. Los parámetros propuestos son valores iniciales de ingeniería, sujetos a una revisión versionada después de medirlos; no son resultados experimentales. No existe un checkpoint nuevo de calidad artística validada en esta entrega.

### 1.1 Lectores y uso

El propietario del proyecto utiliza el SSD para decidir el alcance; los desarrolladores, para implementar interfaces y procesamiento; los responsables de datos, para preparar supervisión; y los evaluadores, para aceptar o rechazar una versión. Los identificadores de requisitos permiten enlazar cambios, pruebas y evidencias.

La fuente editable es `docs/SSD.md`. Una copia PDF debe generarse desde esta fuente sin cambiar el contenido normativo. Las rutas del documento son relativas a la raíz del proyecto, salvo los ejemplos de instalación de Windows.

### 1.2 Convenciones de estado y prioridad

| Código | Significado |
| --- | --- |
| I | Implementado en la base 0.1.0 y contrastado con su código |
| P | Implementación parcial o alcance menor que el objetivo |
| O | Diseño objetivo todavía no implementado |
| P0 | Necesario para una versión funcional que declare control y detalle conservado |
| P1 | Necesario para consistencia por personaje y uso de proyectos extensos |
| P2 | Extensión posterior que no bloquea el primer producto |

## 2 Problemas y resultados esperados

La v2 pública recibe pistas de color, pero su CLI original no las expone. El generador predice colores de forma independiente para cada entrada. Una paleta sin posiciones tampoco indica qué color corresponde a cada personaje o a cada parte de su cuerpo y vestuario.

El issue 39 del repositorio original describe variaciones de pelo y piel entre viñetas y pregunta por pistas y paletas por personaje. El issue 17 describe variaciones cromáticas al cambiar la escala de inferencia y pérdida de detalle al ampliar resultados calculados a menor resolución. Son síntomas diferentes y se evaluarán por separado [R1, R2]. No se atribuye aquí a los autores una causa única demostrada de todos los ejemplos.

### 2.1 Objetivos del producto

- Colorear automáticamente una página B/N y corregirla mediante puntos, trazos y regiones.
- Usar una paleta como preferencia global y especificar colores locales cuando sea necesario.
- Mantener una ficha cromática por personaje y vincularla a regiones concretas de las páginas.
- Calcular el campo de color a una escala de trabajo fija y reutilizarlo en las exportaciones.
- Conservar tinta, texto y detalle fino presentes en el original sin prometer recuperar detalle inexistente.
- Preparar datasets reproducibles, distinguir etiquetas reales de pseudoetiquetas y entrenar el generador extendido.
- Cargar pesos v2 originales de forma estricta y permitir una exportación nativa sin el adaptador global.
- Proporcionar evidencias de funcionamiento y calidad que permitan comparar versiones.

### 2.2 Alcance y exclusiones

La primera versión de producto trabaja localmente con imágenes raster, perfiles cromáticos y máscaras. La segmentación manual es suficiente para su primer flujo de consistencia. La identificación automática de personajes, la segmentación aprendida y el seguimiento entre páginas son extensiones P1 que requieren evaluación y, en su caso, modelos adicionales.

Quedan fuera del primer alcance la edición de texto, traducción, reconstrucción de páginas perdidas, entrenamiento de StyleGAN/pSp/CUT completo de los artículos, generación de dibujos nuevos, vídeo en tiempo real, colaboración en nube y reconocimiento infalible de identidades. La existencia de B/N sin etiqueta no proporciona por sí sola los colores verdaderos.

## 3 Estado de la base y diferencias respecto al objetivo

| Capacidad | Estado | Evidencia o trabajo necesario |
| --- | --- | --- |
| Generador v2 con cinco canales | I | `mangacolor/vendor/v2` y carga estricta |
| Pistas RGB y máscara independiente | I | `inference.py`, normalización y pruebas de negro |
| Pincel y relleno de zonas cerradas | I | Editor HTML, relleno guiado por umbral de tinta |
| Fijación exacta dentro de alfa opaco | I | Composición final posterior a preservar tinta |
| Paleta mediante pistas automáticas | I | Dos pasadas; aproximación sin etiquetas semánticas |
| Adaptador global DeepSets y FiLM | I | Arquitectura, gradientes y serialización comprobados |
| Checkpoint de paleta con calidad validada | O | Entrenamiento y evaluación reservada pendientes |
| Datos sintéticos, parejas y pseudoetiquetas | I | Manifiesto y dataset reproducibles |
| Entrenamiento supervisado y exportación v2 | I | Prueba mecánica y contratos actuales |
| Resolución de salida independiente de inferencia | O | Separar predicción, reconstrucción y exportación |
| Transferencia de color con detalle original | O | Sustituir ampliación RGB completa por reconstrucción |
| Protección explícita de texto y bocadillos | O | Máscaras y prioridad de composición |
| Perfiles por personaje y vínculos regionales | O | Persistencia y asociación manual inicial |
| Consistencia automática entre páginas | O | Segmentación, correspondencias y evaluación |
| Proyectos, lotes y trabajos cancelables | O | El servidor actual es síncrono y procesa una petición |

La salida actual recupera las dimensiones originales mediante Lanczos sobre el RGB predicho. Esto conserva el tamaño, pero no recupera el detalle que la red perdió. `preserve_ink` oscurece los píxeles próximos a tinta; no equivale a la reconstrucción definida en la sección 11.

## 4 Actores y casos de uso

### 4.1 Actores

El **colorista** carga imágenes, pinta pistas, define regiones, revisa predicciones y exporta. El **responsable de datos** registra procedencia y prepara pares. El **entrenador** selecciona checkpoint, datos y configuración. El **evaluador** usa un conjunto reservado y publica un informe. Una persona puede desempeñar todos los papeles; no se exige un sistema de cuentas para el producto local.

### 4.2 Casos de uso principales

**UC01 Colorización y corrección.** El usuario carga una página. El sistema normaliza orientación, muestra el original y genera una propuesta. El usuario añade pistas; la nueva predicción respeta su ubicación. Puede activar un bloqueo exacto explícito y descargar imagen, pistas e informe. Un error conserva original y anotaciones.

**UC02 Exportar a otro tamaño.** Con una propuesta ya calculada, el usuario elige otra resolución. El sistema reconstruye desde el mismo campo canónico sin invocar de nuevo la red. La exportación registra el identificador del campo reutilizado. Una salida mayor no se presenta como superresolución aprendida.

**UC03 Consistencia de personaje.** El usuario crea un perfil, por ejemplo pelo azul y chaqueta roja. Vincula las regiones pertinentes de cada viñeta al perfil. El sistema deriva condiciones locales y registra la versión aplicada. Cambiar el perfil invalida las predicciones afectadas; no modifica silenciosamente imágenes ya exportadas.

**UC04 Preparar datos.** El usuario aporta color y, opcionalmente, B/N alineado. El sistema valida, agrupa, separa y genera un manifiesto. Las imágenes B/N sin objetivo quedan sin etiqueta. Las pseudoetiquetas conservan profesor, peso y estado de revisión.

**UC05 Entrenar y evaluar.** El entrenador inicializa desde v2, ejecuta optimización y guarda estado. El evaluador contrasta el checkpoint con el original en obras reservadas. Solo un informe que satisfaga los criterios aplicables permite declarar mejora de calidad.

**UC06 Exportar a v2.** Se extraen exclusivamente los pesos del generador. Se verifica carga estricta en la arquitectura original. El informe advierte que la paleta global del adaptador no forma parte de ese archivo.

## 5 Requisitos funcionales

### 5.1 Entrada y anotación

| ID | Prioridad | Requisito | Estado |
| --- | --- | --- | --- |
| IN01 | P0 | Leer PNG, JPEG, WebP, BMP y TIFF compatibles con Pillow; documentar el subconjunto del editor | P |
| IN02 | P0 | Aplicar orientación EXIF y componer transparencias sobre blanco sin cambiar coordenadas después | I |
| IN03 | P0 | Validar dimensiones, formato y límites antes de inferencia; conservar original inmutable | P |
| IN04 | P0 | Pistas y máscaras deben corresponder al original ya orientado | I |
| LOC01 | P0 | Distinguir negro, blanco y transparencia mediante un canal de presencia | I |
| LOC02 | P0 | Proporcionar pincel continuo, borrador, deshacer y relleno revisable | I |
| LOC03 | P0 | Separar predicción guiada y composición exacta en la interfaz y en los informes | I |
| LOC04 | P0 | Conservar las anotaciones de resolución original al cambiar la vista o exportar | P |
| LOC05 | P1 | Guardar y reabrir capas de pistas y máscaras en un proyecto | O |

### 5.2 Paleta y consistencia

| ID | Prioridad | Requisito | Estado |
| --- | --- | --- | --- |
| PAL01 | P0 | Aceptar colores HEX sRGB y validar valores finitos y límites | I |
| PAL02 | P0 | Etiquetar el modo nativo de paleta como aproximación mediante pistas | I |
| PAL03 | P0 | Rechazar paleta aprendida cuando el checkpoint no registra actualizaciones del adaptador | I |
| PAL04 | P0 | La paleta global aprendida debe ser invariante al orden y a entradas enmascaradas | I |
| PAL05 | P0 | Informar si la aproximación nativa coloca cero muestras cromáticas | P |
| CON01 | P1 | Guardar perfiles de personaje con partes, colores base y referencias | O |
| CON02 | P1 | Vincular máscaras con personaje y parte de forma explícita | O |
| CON03 | P1 | Propagar una revisión del perfil solo tras regenerar los resultados afectados | O |
| CON04 | P1 | Permitir variación de iluminación manteniendo identificable el color base | O |
| CON05 | P2 | Sugerir correspondencias automáticas sin sobrescribir vínculos humanos | O |

### 5.3 Resolución y trazabilidad

| ID | Prioridad | Requisito | Estado |
| --- | --- | --- | --- |
| RES01 | P0 | Separar escala de inferencia y dimensiones de exportación | O |
| RES02 | P0 | Reutilizar exactamente el mismo campo canónico al cambiar solo la exportación | O |
| RES03 | P0 | Reconstruir color con guía del original y detalle fino conservado | O |
| RES04 | P0 | Proteger píxeles de una máscara de texto suministrada y validada | O |
| RES05 | P0 | No atribuir invariancia de la red a cambios reales de escala de inferencia | P |
| RES06 | P0 | Registrar espacio de color, perfil de preprocesado y versión de reconstrucción | O |
| OUT01 | P0 | Exportar PNG e informe sin sobrescribir entradas ni resultados existentes | P |
| OUT02 | P1 | Exportar proyectos y lotes reanudables con manifiesto de resultados | O |
| OUT03 | P0 | Registrar hashes de imagen, modelo, condiciones y configuración | P |

Los requisitos de datos, entrenamiento, compatibilidad y operación se desarrollan en sus secciones para conservar una única definición de cada contrato.

## 6 Requisitos no funcionales

| ID | Requisito verificable | Estado |
| --- | --- | --- |
| NF01 | El procesamiento normal se ejecutará localmente sin enviar imágenes a servicios externos | I |
| NF02 | La ejecución CPU debe estar disponible; CUDA será opcional y se validará explícitamente | I |
| NF03 | Un checkpoint incompatible debe fallar antes de producir una imagen | I |
| NF04 | Los originales deben permanecer inmutables en todos los casos de uso | P |
| NF05 | Las versiones de modelo, datos, preprocesado y reconstrucción deben poder identificarse | P |
| NF06 | Los resultados deterministas se exigirán dentro de un entorno y precisión fijados, no entre cualquier GPU | P |
| NF07 | Las escrituras finales de proyecto, resultado y checkpoint deben ser atómicas | P |
| NF08 | La interfaz objetivo deberá aceptar cancelación entre etapas y mostrar errores recuperables | O |
| NF09 | No habrá ampliación automática de exposición desde loopback a red local | I |
| NF10 | Las métricas de rapidez y memoria se medirán por hardware, resolución y modo | P |
| NF11 | No se declarará calidad artística a partir de pruebas unitarias o de tres pasos de entrenamiento | P |
| NF12 | El usuario podrá inspeccionar los parámetros y evidencias asociados a un resultado | P |
| NF13 | La aplicación objetivo distinguirá advertencia de calidad, entrada inválida y fallo interno | O |
| NF14 | Las dependencias de cada versión publicada deberán fijarse en un archivo reproducible | O |

No se fija todavía un tiempo máximo universal por página: no hay una medición representativa a resolución de producción. El objetivo de exportación es evitar el coste de otra inferencia; su latencia y memoria siguen creciendo con los píxeles de salida.

## 7 Arquitectura del sistema

### 7.1 Componentes y responsabilidades

| Componente | Responsabilidad | Ubicación actual o propuesta |
| --- | --- | --- |
| CLI | Validación de argumentos y ejecución de comandos | `mangacolor/cli.py` |
| Editor | Carga de página, pincel, relleno y descarga | `mangacolor/web/editor.html` |
| API local | Decodificación PNG y acceso al pipeline | `mangacolor/editor.py` |
| Pipeline | Preprocesado, condiciones e inferencia | `mangacolor/inference.py` |
| Modelo | Generador v2 y adaptador global | `mangacolor/model.py` |
| Pesos | Lectura estricta y exportación | `mangacolor/checkpoints.py` |
| Datos | Preparación y muestreo de supervisión | `mangacolor/data.py` |
| Entrenamiento | Pérdidas, optimización y validación | `mangacolor/training.py` |
| Reconstrucción | Ampliación guiada, luminosidad y composición | Objetivo `mangacolor/reconstruction.py` |
| Proyectos | Perfiles, vínculos, revisiones y persistencia | Objetivo `mangacolor/projects.py` |
| Trabajos y caché | Cola, cancelación y reutilización del campo | Objetivo `mangacolor/jobs.py` y `cache.py` |
| Evaluación | Comparaciones reservadas e informe de calidad | Objetivo `mangacolor/evaluation.py` |

Las ubicaciones propuestas no son importaciones disponibles. El generador original copiado en `vendor/v2` conserva nombres y formas de parámetros; no debe alterarse inadvertidamente durante refactorizaciones de servicio.

### 7.2 Flujo de inferencia objetivo

```text
Original y anotaciones
        |
        v
Normalización y coordenadas originales
        |
        v
Entrada canónica y condiciones de color
        |
        v
Caché de predicción ----> Generador v2 y adaptador opcional
        |                              |
        +--------- Campo canónico <----+
                         |
Original completo ------+----> Reconstrucción guiada
Máscaras protegidas ---------> Composición por prioridades
                                      |
                                      v
                           PNG e informe de exportación
```

La reconstrucción no debe depender de una llamada oculta a la red. La predicción canónica y la exportación son operaciones distintas y deben poder probarse con un contador de invocaciones del modelo.

### 7.3 Flujo de entrenamiento

```text
Color o pares auténticos -> Validación -> Agrupación y separación
                                                |
                                                v
                                           Manifiesto
                                                |
B/N sin etiqueta -> Profesor -> Revisión -> Pseudoentrenamiento
                                                |
                                                v
                            Dataset -> Modelo -> Pérdidas -> Checkpoint
                                                          |
Obras reservadas ----------------------------------------> Evaluación
```

Las obras reservadas no se incorporan al entrenamiento a través del profesor ni de variantes sintéticas.

## 8 Modelo de dominio y persistencia objetivo

### 8.1 Entidades

**Proyecto** contiene páginas, perfiles de personajes, configuración y revisiones. **Página** referencia un original inmutable ya orientado, viñetas y anotaciones. **Región** contiene una máscara y su significado, sin imponer que el detector automático sea la fuente. **Perfil** define colores base por parte del personaje. **Predicción** almacena el campo canónico. **Exportación** identifica una reconstrucción y un archivo concreto. **Dataset** y **ejecución de entrenamiento** son entidades independientes de un proyecto de colorización.

Los identificadores objetivo son cadenas opacas estables; no se usarán nombres de personajes como claves técnicas. La versión de esquema debe comprobarse al abrir cualquier proyecto. La migración crea una copia recuperable y nunca altera el original silenciosamente.

### 8.2 Coordenadas y máscaras

Las coordenadas persistidas utilizan el original después de EXIF, con origen superior izquierdo. Los píxeles se indexan con enteros `x` e `y`; las cajas son intervalos semiabiertos `[x0, y0, x1, y1)`. Los trazos vectoriales pueden guardar puntos normalizados en `[0,1]`, pero su rasterización debe incluir dimensiones, grosor en píxeles originales y versión del rasterizador.

Las máscaras raster son PNG de un canal o alfa de PNG RGBA, con el tamaño del original. Cero significa ausencia y 255 cobertura completa. La transformación a escala canónica se conserva en el informe. La compresión JPEG no debe emplearse para máscaras.

### 8.3 Esquema de proyecto propuesto

```json
{
  "schema": "mangacolor.project.v1",
  "project_id": "project_001",
  "revision": 3,
  "color_space": "sRGB",
  "pages": [{
    "page_id": "page_001",
    "original": "assets/page_001.png",
    "pixel_sha256": "<sha256 de pixeles normalizados>",
    "width": 1125,
    "height": 1600,
    "regions": [{
      "region_id": "region_001",
      "mask": "masks/page_001_hair.png",
      "character_id": "character_001",
      "part_id": "hair",
      "profile_revision": 2,
      "mode": "base_color",
      "priority": 100
    }]
  }],
  "characters": ["profiles/character_001.json"],
  "render_config": "config/render.json"
}
```

Los textos entre ángulos son descripciones de campos, no hashes válidos. Los ejemplos de esquemas objetivo no los acepta todavía la aplicación 0.1.0.

### 8.4 Estructura de almacenamiento propuesta

```text
project/
  project.json
  assets/              originales importados y normalizados
  masks/               regiones y protecciones
  annotations/         trazos y pistas RGBA
  profiles/            personajes y referencias
  config/              procesamiento y modelos
  predictions/         campos canonicos sin pickle
  exports/             imagenes e informes
  logs/                eventos y errores
```

Los archivos grandes de caché pueden residir fuera del paquete de proyecto y reconstruirse. Los originales, perfiles y máscaras no son caché y no se eliminan al limpiarla.

## 9 Contratos de imagen y preprocesado

### 9.1 Entrada actual

`open_rgb` aplica EXIF, convierte a RGBA, compone sobre blanco y devuelve RGB. La inferencia obtiene gris mediante `Pillow.convert('L')`; esta conversión de trabajo no es idéntica a luminancia lineal ni a CIELAB L*. No debe mezclarse su escala con las fórmulas de reconstrucción sin conversión explícita.

Con `size=S`, la base actual fija el lado mayor:

```text
r = S / max(W_original, H_original)
W = max(1, round(r * W_original))
H = max(1, round(r * H_original))
pad_right  = (-W) mod 32
pad_bottom = (-H) mod 32
```

Se usa Lanczos para el gris, padding blanco para imagen y cero para pistas. El padding se retira antes de ampliar la salida. `S` debe estar entre 32 y 2048 en la API del pipeline. El valor 576 de esta reescritura se refiere al lado mayor; el preprocesado original usa otra política. No se deben comparar ambos recorridos como si fueran idénticos.

### 9.2 Pistas actuales

El RGB de una pista se normaliza con `2*RGB/255 - 1` y se multiplica por alfa. Se redimensiona el conjunto premultiplicado; al reducir se usa interpolación de área y al aumentar bilineal. Así una pista negra tiene RGB negativo y máscara positiva, mientras que ausencia de pista tiene los cuatro canales a cero.

El redimensionado puede reducir el alfa de un trazo de un píxel y, por tanto, su fuerza. No se garantiza una respuesta idéntica a distintas escalas. El objetivo es permitir también un grosor mínimo canónico configurable, registrando cuándo una pista se amplía para mantenerla visible.

### 9.3 Política de color objetivo

La salida de intercambio será PNG sRGB de 8 bits por canal. El procesamiento de reconstrucción utilizará `float32`. Las entradas con perfil ICC deben convertirse a sRGB mediante gestión de color antes de calcular hashes de píxeles; si no se puede interpretar el perfil, el sistema objetivo advertirá y requerirá una política explícita de asumir sRGB o rechazar. Esta gestión ICC completa todavía no está implementada.

La entrada CMYK, imágenes de 16 bits, TIFF multipágina y GIF animado requieren una política documentada. El producto inicial debe seleccionar una página o fotograma explícito; no debe descartar silenciosamente páginas o precisión. Los límites de formatos que Pillow abre no constituyen por sí solos una garantía de todas estas variantes.

## 10 Arquitectura del modelo y control cromático

### 10.1 Contrato tensorial

| Tensor | Forma | Rango y significado |
| --- | --- | --- |
| `input` | B x 5 x H x W | Gris, tres pistas RGB enmascaradas y máscara |
| Gris | B x 1 x H x W | Float32 en `[0,1]` |
| Pistas RGB | B x 3 x H x W | RGB normalizado `[-1,1]` multiplicado por máscara |
| Máscara | B x 1 x H x W | Cobertura `[0,1]` |
| `palette` | B x K x 3 | RGB normalizado `[-1,1]` |
| `palette_mask` | B x K | Entradas binarias 0 o 1 |
| Salida principal | B x 3 x H x W | RGB con tanh en `[-1,1]` |
| Salida auxiliar | B x 3 x h x w | Guía de entrenamiento; no se usa como exportación |

H y W deben ser múltiplos de 32. El servicio debe validar finitud y límites antes de crear tensores grandes. La validación de formas está implementada en el modelo; no todos los límites del servicio objetivo se aplican hoy a llamadas Python directas.

### 10.2 Base v2

La primera convolución de la rama de pistas tiene forma `(32,5,3,3)`. `to0` produce 32 canales a resolución completa. La rama auxiliar reduce mediante `to1`, `to2` y `to3` a 64, 92 y 128 canales. El extractor SEResNeXt recibe solo gris y produce características multiescala que se fusionan en los túneles del decodificador.

Los niveles principales del extractor tienen 64, 256, 512 y 1024 canales; sus escalas efectivas son H/2, H/2, H/4 y H/8. Los túneles fusionan estructura y pistas, utilizan bloques ResNeXt y PixelShuffle, y la salida principal aplica tanh. La arquitectura conserva también módulos registrados no recorridos por el `forward` actual, como `to4` y `tunnel1`, porque sus parámetros pertenecen al contrato de los checkpoints originales.

No se cargan pesos v1 en esta arquitectura. El generador v1 utiliza seis canales y un campo adicional de distancia a líneas. El sistema completo con borradores latentes de los artículos de 2023 tampoco equivale al generador público de cinco canales.

### 10.3 Adaptador de paleta

Cada color pasa por un MLP compartido `3 -> 64 -> 64` con SiLU. Se calcula la media de características de entradas válidas y una proyección `64 -> 64` produce escala y desplazamiento para los 32 canales de `x0`:

```text
z = sum_k(m_k * encoder(p_k)) / max(1, sum_k(m_k))
[gamma, beta] = projection(z) * indicator(sum_k(m_k) > 0)
x0_control = x0 * (1 + gamma) + beta
```

La proyección se inicializa a cero. Por ello, la arquitectura extendida reproduce la salida base antes de aprender el adaptador, bajo el mismo tensor, pesos y entorno. Una máscara completamente vacía mantiene la modulación nula incluso después de aprender un sesgo.

La representación es invariante al orden del conjunto dentro de tolerancia numérica. No codifica identidad de personaje ni asigna partes semánticas a colores. Los colores duplicados se enmascaran en el dataset; no se debe utilizar su repetición como forma implícita de expresar pesos.

### 10.4 Modos globales disponibles

**Modo `hints`.** Ejecuta una propuesta base, toma muestras espaciadas y asigna colores de la paleta por proximidad cromática tras restar la media RGB. Evita negro intenso, muestras poco cromáticas y píxeles con pistas manuales. Ejecuta una segunda pasada. No es una cuantización final ni un adaptador entrenado; puede colocar cero muestras y no garantiza asignación semántica correcta.

**Modo `learned`.** Alimenta el adaptador con la paleta. La carga ordinaria bloquea su uso si `palette_trained` es falso. El entrenamiento puede habilitar la rama sin entrenar para aprenderla. La bandera registra procedencia de optimización; no acredita calidad.

La CLI limita la paleta global a 32 colores; el dataset usa 8 por defecto. La arquitectura acepta conjuntos variables, pero entrenar con K=8 no demuestra generalización a cualquier tamaño de paleta. Debe evaluarse por separado.

### 10.5 Condiciones locales y bloqueos

Una pista guía la predicción de la red y puede influir fuera de su máscara. El bloqueo exacto es una composición posterior. La versión actual aplica primero `preserve_ink` y después el bloqueo; un píxel con alfa 255 se fija exactamente al RGB solicitado incluso si cubre tinta. Un alfa intermedio mezcla y no garantiza igualdad exacta.

El modo objetivo `base_color` preserva iluminación y detalle; el modo `exact_rgb` prioriza el valor final. La interfaz no debe utilizar el mismo nombre para ambos comportamientos.

## 11 Reconstrucción a alta resolución objetivo

Esta sección es un diseño pendiente. Puede desarrollarse sobre los pesos v2 actuales sin reentrenarlos. No mejora por sí sola una decisión semántica equivocada de color.

### 11.1 Separación de escalas

El perfil inicial propuesto es `canonical_long_edge_v1`, con lado mayor 576 y padding a múltiplos de 32. Se conservará un perfil de compatibilidad explícito para reproducir el preprocesado original cuando sea necesario; no se cambiarán ambos bajo el mismo identificador.

La resolución de exportación se expresará como `original` o como dimensiones de salida con aspecto conservado. Cambiarla no altera el tensor de inferencia, la paleta, las pistas, la semilla ni el checkpoint. Cambiar la escala canónica constituye una nueva predicción y puede modificar sus colores.

No se ofrece identidad exacta entre archivos de entrada que hayan sido remuestreados externamente. Para comparar tamaños de salida se usará el mismo original normalizado y el mismo campo de color almacenado.

### 11.2 Descomposición y espacio de color

La propuesta RGB canónica se convierte desde sRGB a RGB lineal y después a XYZ y CIELAB con blanco D65. El original gris se trata mediante una conversión coherente al mismo espacio. Debe fijarse la implementación y probarse con parches de referencia.

Para un componente sRGB `c` en `[0,1]`:

```text
c_linear = c / 12.92                         si c <= 0.04045
c_linear = ((c + 0.055) / 1.055) ** 2.4      en otro caso
```

Se empleará la matriz sRGB D65 estándar y el blanco XYZ `(0.95047, 1.00000, 1.08883)` con XYZ escalado a Y=1. Los rangos de trabajo son L* en `[0,100]` y a*, b* firmados. No se interpretarán canales Lab de una librería en 8 bits como si ya estuvieran en estas unidades.

### 11.3 Ampliación guiada

Los canales a* y b* se amplían con un filtro guiado por la estructura del original. El filtro debe ser una operación local con parámetros registrados, por ejemplo radio proporcional a la escala y regularización normalizada. Su propósito es reducir sangrado y halos; no sustituye una segmentación de objetos [R7].

La primera implementación debe comparar interpolación bilineal con filtro guiado bajo las mismas condiciones. Si el filtro introduce textura de trama en el color, se utilizará una guía suavizada o protección regional. El filtrado por tiles requiere un margen de contexto al menos igual al soporte efectivo; se recortará ese margen al unir para evitar costuras.

### 11.4 Luminosidad y detalle

Copiar literalmente la luminosidad de una superficie blanca impediría representar muchos colores saturados sin salir de gama. Por ello se preservan los detalles finos del original y se permite que la predicción determine la iluminación suave.

Un algoritmo inicial reproducible es:

```text
L_original = luminosidad del original en resolucion completa
B_original = filtro_suave_versionado(L_original)
D_original = L_original - B_original
B_predicho = ampliar_suavemente(L_predicho_canonico)
L_mezclado = clip(B_predicho + detalle_strength * D_original, 0, 100)
L_final = alpha_original * L_original + (1-alpha_original) * L_mezclado
```

`detalle_strength` y `alpha_original` son parámetros objetivo, no flags actuales. Sus valores iniciales propuestos son 1.0 y 0.0 respectivamente para conservar detalle con iluminación predicha; se ofrecerá un modo de luminosidad original para quien lo prefiera. La evaluación deberá impedir que un aumento de detalle amplifique ruido, tramas o contornos con halos.

No se afirmará que esta descomposición conserva exactamente todos los píxeles de tinta o texto. La igualdad de píxeles protegidos procede de máscaras explícitas y composición final, no de la aproximación de frecuencias.

### 11.5 Gama de color

Al recomponer Lab con una luminosidad distinta, algunos valores pueden quedar fuera de sRGB. El algoritmo objetivo reducirá gradualmente la cromaticidad en dirección al eje neutro a luminosidad fijada hasta entrar en gama, y registrará la fracción de píxeles afectados. El recorte por canal puede utilizarse como última defensa numérica, pero no será el único método de mapeo de gama.

Se rechazarán NaN e infinitos. Antes de codificar se verificará `[0,1]`, se aplicará la función inversa sRGB y se cuantizará de forma determinista. Los cambios inevitables por gamut y remuestreo se distinguirán de una nueva decisión de la red.

### 11.6 Prioridades de composición

La política objetivo predeterminada preserva texto y tinta explícitamente protegidos. El orden será reconstrucción, corrección regional de color base, bloqueos exactos permitidos y restauración de píxeles protegidos. La máscara protegida siempre prevalece, salvo que el usuario desactive explícitamente esa protección para una región y esa decisión quede registrada.

Esto difiere del comportamiento 0.1.0, donde el bloqueo exacto es lo último. La migración deberá identificar el perfil de composición para reproducir resultados anteriores. Los conflictos deben mostrarse antes de exportar y no resolverse silenciosamente.

Para garantía exacta, las máscaras de texto/tinta protegidos son binarias y se restauran desde el original a resolución original. El antialiasing de límites se aplica fuera del interior protegido. Si se exporta a otra resolución, la comparación será contra el original remuestreado con el mismo filtro declarado; no se prometerá identidad entre tamaños diferentes.

### 11.7 Campo canónico y caché

Se guardarán RGB o Lab canónico en float32, transformaciones, dimensiones y metadatos en NPZ sin objetos y JSON, con `allow_pickle=False`. El campo es un artefacto reutilizable; no se recargará mediante pickle general.

La clave de predicción incluirá hashes de píxeles originales normalizados, checkpoint completo, tensor de condiciones canónicas, paleta y máscara, perfil de preprocesado, versión de código de inferencia, modo de paleta, precisión y entorno determinista. Los perfiles de personaje se rasterizan a condiciones y se registra además su revisión. Las dimensiones de exportación quedan excluidas.

La clave de exportación añade la clave de predicción, dimensiones finales, versión y parámetros de reconstrucción, máscaras protegidas, bloqueos y formato. Un cambio de protección puede requerir solo reconstrucción; un cambio de pista o paleta requiere nueva predicción.

La caché objetivo tendrá cuota configurable y expulsión LRU de artefactos regenerables. No se fija una cuota universal antes de medir almacenamiento; se mostrará el consumo y se conservarán originales y anotaciones al limpiarla.

## 12 Consistencia por personaje objetivo

### 12.1 Perfil cromático

```json
{
  "schema": "mangacolor.character.v1",
  "character_id": "character_001",
  "name": "Personaje A",
  "revision": 2,
  "parts": {
    "hair": {"base_srgb": "#285DAD", "mode": "base_color"},
    "skin": {"base_srgb": "#DBA98A", "mode": "base_color"},
    "jacket": {"base_srgb": "#A63835", "mode": "base_color"}
  },
  "references": ["references/character_001_front.png"]
}
```

Los colores base son preferencias editables; no representan el RGB final de cada sombra. Cambios de ropa se modelan como variantes del perfil o conjuntos de partes, no como excepciones escondidas. La visibilidad y la oclusión pertenecen a cada región de página.

### 12.2 Primera implementación manual

El usuario dibuja o rellena máscaras para pelo, piel y ropa, y las vincula a una parte del perfil. El sistema produce pistas locales desde el color base y conserva máscara y vínculo en el proyecto. Se debe mostrar una superposición para comprobar el alcance antes de inferir.

Un objeto sin vínculo mantiene colorización libre. Si dos regiones se superponen, se resuelve por prioridad explícita y, en empate, por orden de capa persistido. No se mezclarán dos identidades por una heurística no visible. Las máscaras de protección se procesan según la sección 11.6.

### 12.3 Automatización posterior

La segmentación y la correspondencia automática proponen máscaras e identidades con confianza y procedencia. El usuario podrá corregirlas y fijarlas. Una propuesta no debe reemplazar anotaciones revisadas. La aplicación debe admitir el estado desconocido cuando hay ambigüedad, personajes similares u oclusión.

Para entrenar consistencia aprendida se necesitan grupos de apariciones y correspondencias de partes. Una pérdida entre dos viñetas solo es válida donde exista correspondencia confiable, con exclusión de sombras extremas y cambios de vestuario. Los pares no se generan suponiendo que todo el pelo de todas las páginas pertenece a la misma identidad.

Una extensión posible añade condiciones regionales o referencias al modelo. Si cambia el número de canales o la arquitectura, tendrá un formato de checkpoint nuevo y un conversor explícito; no se declarará compatible mediante carga parcial.

## 13 Preparación de datasets

### 13.1 Fuentes y requisitos

| ID | Requisito | Estado |
| --- | --- | --- |
| DAT01 | No modificar imágenes fuente durante preparación | I |
| DAT02 | Registrar objetivo real, sintético, pseudo o ausencia de etiqueta | I |
| DAT03 | Mantener variantes y duplicados exactos en la misma partición | I |
| DAT04 | Separar obras completas cuando la estructura de fuentes lo permita | I |
| DAT05 | Comprobar tamaño de parejas y exigir revisión del alineamiento real | P |
| DAT06 | No usar pseudoetiquetas como verdad de referencia en validación | I |
| DAT07 | Guardar fuente, permiso de uso y versión de colección | P |
| DAT08 | Añadir un conjunto de test inmutable separado de train y val | O |
| DAT09 | Detectar y revisar duplicados próximos además de hashes exactos | O |
| DAT10 | Permitir supervisión parcial mediante máscara de validez del objetivo | O |

El preparador recorre formatos raster admitidos. Considera una imagen de color si más del 0,1 % de sus píxeles presentan diferencia entre canales mayor que 10 y saturación relativa mayor que 0,08. Es una heurística para evitar ruido cromático de compresión, no una etiqueta semántica infalible. Las clasificaciones dudosas deben poder revisarse en la versión objetivo.

### 13.2 Construcción actual

De una fuente color se guarda el objetivo RGB y dos entradas sintéticas: luminancia y tinta aproximada mediante filtros Pillow. No es XDoG ni la traducción aprendida de dominio de los artículos. Si hay un directorio B/N, se busca la misma ruta relativa sin extensión, ignorando diferencias de mayúsculas. Se rechaza una coincidencia ambigua y una pareja de dimensiones distintas.

El hash de agrupación usa dimensiones y píxeles RGB normalizados. El campo `sha256` identifica los bytes del archivo fuente. Las variantes tienen nombres derivados del hash, y los duplicados exactos no multiplican su peso por aparecer con nombres distintos.

Con al menos dos subcarpetas superiores de color, se agrupa por esa carpeta como obra; los grupos se unen si comparten píxeles duplicados. Con una sola obra se separa por hash de imagen. En ambos casos todas las variantes permanecen juntas. Con un único grupo se conserva entrenamiento y no se inventa una validación.

La fracción de validación es 0,1 y la semilla 42 por defecto. Si existe más de un grupo y una fracción positiva, se reserva al menos uno sin vaciar entrenamiento. Los resultados dependen del inventario completo y deben regenerarse bajo una versión nueva si cambia la colección.

### 13.3 Manifiesto actual

Cada línea UTF-8 JSONL contiene un registro. `input` y `target` se resuelven respecto al manifiesto; después de pseudoetiquetar pueden existir rutas absolutas a datos previos.

| Campo | Tipo | Significado |
| --- | --- | --- |
| `id` | string | Identificador de muestra y variante |
| `input` | string | Imagen B/N de entrada |
| `target` | string o null | Objetivo RGB o ausencia de etiqueta |
| `split` | string | `train`, `val` o `unlabeled` |
| `supervision` | string | `synthetic`, `paired`, `pseudo` o `unlabeled` |
| `variant` | string | Luminancia, tinta, pareja o no etiquetada |
| `source` | string | Ruta relativa de la fuente |
| `sha256` | string | SHA256 de bytes de la fuente |
| `group` | string | Hash de dimensiones y píxeles |
| `source_group` | string o null | Carpeta superior de origen |
| `split_group` | string | Grupo efectivo indivisible |
| `width`, `height` | integer | Dimensiones tras orientación |
| `weight` | number opcional | Peso positivo finito; pseudo por defecto 0,25 |
| `teacher` | object opcional | Metadatos del profesor de pseudoetiquetas |

El preparador no permite un destino dentro de las fuentes. Un manifiesto existente diferente provoca error antes de crear nuevas imágenes. Una ejecución idéntica es idempotente. Los PNG generados existentes se comparan y no se sobrescriben con contenido distinto. La escritura final del manifiesto utiliza archivo temporal y reemplazo.

El cargador rechaza fuga train/val por identificador, grupo, grupo de separación y rutas resueltas de entrada/objetivo. Los hashes exactos no detectan todas las copias remuestreadas, recortadas o recomprimidas; DAT09 completa esa limitación.

### 13.4 B/N y pseudoetiquetas

Las imágenes monocromas se registran con `target=null` y no participan en la pérdida supervisada. El comando `pseudo-label` ejecuta el profesor sobre las no reservadas y genera objetivos con `supervision=pseudo`, `split=train`, `weight=0.25` y metadatos del profesor, incluido el hash del checkpoint.

No se pseudoetiquetan registros de validación ni grupos asociados a obras reservadas según `source_group`, `split_group` o `group`. El objetivo producido es una hipótesis. El flujo objetivo debe añadir `review_status`, anotador y revisión; los objetivos no revisados conservarán su procedencia aun cuando se corrijan parcialmente.

Un B/N con unas pocas pistas no puede convertirse en un objetivo RGB completo sin inventar las zonas restantes. DAT10 requiere una máscara adicional de validez y pérdidas restringidas al área supervisada. No existe aún ese modo de entrenamiento parcial en la base.

### 13.5 Dataset objetivo ampliado

Se añadirá una versión de esquema con `dataset_id`, licencia o autorización registrada, `work_id`, `chapter_id`, `page_id`, `character_ids`, tipo de etiqueta, transformación de alineamiento y revisión. Las particiones train/val/test se congelarán por obra antes de crear variantes o pseudoetiquetas. Las estadísticas por fuente y estilo se guardarán junto al manifiesto.

El test incluirá tinta limpia, tramas, líneas finas, texto pequeño, páginas muy altas, varios personajes, sombras fuertes y estilos no presentes en entrenamiento. La cantidad necesaria depende de la variabilidad de la colección; se propone como mínimo operativo inicial 200 páginas de al menos cinco obras para una evaluación de lanzamiento, etiquetado como criterio propuesto y no como dataset ya disponible.

## 14 Muestreo y aumentos de entrenamiento

`PaletteDataset` devuelve los tensores de la sección 10 y un identificador. Usa un tamaño cuadrado múltiplo de 32, por defecto 256. Reescala manteniendo aspecto hasta cubrir el recorte y aplica las mismas transformaciones geométricas a entrada y objetivo.

En entrenamiento el recorte y el espejo horizontal son aleatorios reproducibles. Se modifica el objetivo con recoloración HSV global con probabilidad 0,5, conservando después la coherencia entre objetivo, pistas y paleta. No es recoloración regional por personaje. La validación usa recorte central y no recolorea el objetivo.

Las pistas se generan con ausencia total en el 20 % de muestras. En las restantes se combinan puntos, brochazos y regiones conectadas de color parecido en el objetivo. Las regiones no se construyen como rectángulos indiscriminados sobre distintos objetos. La paleta se extrae por cuantización Median Cut de una miniatura hasta 256 x 256, se ordena por frecuencia y se rellena hasta K; la máscara marca únicamente colores distintos. Se elimina toda la paleta con probabilidad 0,2.

La semilla de muestreo depende de semilla global, época e índice. `set_epoch` cambia los aumentos de entrenamiento; en validación la época no altera el resultado. Esto no convierte toda la ejecución CUDA en determinista entre entornos diferentes.

El objetivo completo de validación debe contener color según la heurística; un recorte que casualmente sea blanco sigue siendo válido. Las dimensiones de la pareja se verifican antes de transformar. La evaluación objetivo de páginas completas se hará fuera del recorte central para evitar ignorar texto y bordes.

## 15 Entrenamiento y optimización

### 15.1 Alcances actuales

**`decoder`.** Congela los parámetros cuyo nombre comienza por `encoder.` y ajusta el resto del generador y el adaptador. Mantiene el extractor en modo evaluación para evitar alterar sus estadísticas con lotes pequeños. Este alcance permite mejorar la respuesta a pistas locales.

**`adapter`.** Congela todos los pesos base y ajusta solo `palette_adapter.`. El generador permanece en evaluación y el adaptador en entrenamiento. Este alcance enseña la condición global, pero no modifica por sí mismo los pesos que propagan pistas locales.

La optimización actual es AdamW, tasa inicial 0,00002 y parámetros restantes predeterminados de PyTorch. Se recorta la norma global del gradiente a 1. La rama CUDA usa autocast y GradScaler; CPU no utiliza precisión mixta. Los fallos por NaN o gradiente no finito detienen la ejecución en vez de guardar pesos defectuosos.

### 15.2 Pérdidas implementadas

Para una imagen i, `P` es la salida principal, `T` el objetivo normalizado, `G` la guía y `M` la máscara de pistas. La guía se compara con el objetivo reducido a su tamaño mediante interpolación de área.

```text
L_rgb(i)   = mean(abs(P_i - T_i))
L_guide(i) = mean(abs(G_i - resize_area(T_i)))
L_hint(i)  = sum(abs(P_i-T_i) * M_i) / max(1, 3 * sum(M_i))
```

La máscara se normaliza por área de pista para que un punto no quede diluido entre todos los píxeles. Si no hay pistas su contribución es cero.

La pérdida global usa la salida agrupada a 16 x 16, resta la media de sus canales RGB y compara esos vectores cromáticos con los de las entradas válidas de paleta. Para cada posición toma el mínimo error cuadrático medio entre colores y después promedia posiciones. Si la paleta está vacía, la contribución es cero.

```text
L_palette(i) = mean_j(min_k mean_channels((C_pred(i,j)-C_palette(i,k))^2))
L_total(i) = L_rgb(i) + 0.3*L_guide(i) + 2*L_hint(i) + 0.1*L_palette(i)
L_batch = mean_i(weight_i * L_total(i))
```

La pérdida de paleta es un objetivo auxiliar RGB centrado, no una distancia perceptual Lab ni un mecanismo de identidad. No existe aquí una pérdida adversarial, VGG perceptual, de bordes, de consistencia entre escalas o entre personajes. No se deben incluir sus efectos en afirmaciones sobre el entrenamiento actual.

El peso pseudo de 0,25 multiplica la contribución antes de promediar por el tamaño del lote, no por la suma de pesos. Con optimizadores adaptativos, reducir una magnitud de gradiente no implica una reducción lineal idéntica del desplazamiento final de parámetros; se medirá el efecto real de la mezcla de fuentes.

### 15.3 Mejoras de entrenamiento objetivo

| ID | Requisito | Estado |
| --- | --- | --- |
| TRN01 | Inicializar estrictamente desde v2 o desde un checkpoint extendido válido | I |
| TRN02 | Registrar hiperparámetros, pérdidas y tamaños de conjuntos | I |
| TRN03 | Guardar actualizaciones reales de paleta y separar entrenamiento de calidad validada | I |
| TRN04 | Evaluar sin pistas, con pistas y con paleta por separado | P |
| TRN05 | Añadir validación a página completa y métricas de detalle | O |
| TRN06 | Añadir datos y pérdida entre correspondencias del mismo personaje | O |
| TRN07 | Añadir consistencia multiescala si se quiere robustez al cambiar inferencia real | O |
| TRN08 | Guardar estado suficiente para reanudación exacta cuando se declare esa garantía | O |
| TRN09 | Guardar hash de manifiesto, código, entorno y política de preprocesado | P |
| TRN10 | Comparar exportación nativa y modelo extendido en el test reservado | O |

Una pérdida multiescala compararía predicciones del mismo original llevadas a coordenadas comunes, con condiciones equivalentes. Se excluirán remuestreos ambiguos y zonas sin correspondencia. No hace falta esa pérdida para reutilizar un campo al cambiar únicamente la resolución de exportación.

La incorporación de pérdidas perceptuales o adversariales requerirá una ablación: reconstrucción, control local, fidelidad de perfil, detalle y estabilidad sin pistas. No se añadirá una GAN solo por reproducir el nombre del método original si degrada las garantías del producto.

### 15.4 Reanudación y artefactos actuales

Se guardan `last.pt`, `config.json` y `metrics.jsonl`. La bandera `palette_trained` solo se activa tras un lote con paleta válida y gradiente no nulo del adaptador. `quality_validated` permanece falso y `training_status` es experimental.

`--resume` restaura modelo, optimizador, número de época y pasos; exige el mismo alcance. Empieza una época nueva. No restaura cursor de lote, RNG completo ni GradScaler, y no verifica todavía igualdad del manifiesto. `--epochs` expresa épocas adicionales. Esta es una reanudación operativa, no reproducción exacta de un paso interrumpido.

El objetivo es almacenar también hash del manifiesto, configuración inmutable, estado AMP, RNG y cursor, y escribir checkpoint temporal seguido de reemplazo atómico. Cambiar tamaño, datos o alcance se declarará como una nueva ejecución derivada.

## 16 Checkpoints y compatibilidad

### 16.1 Formatos disponibles

El archivo `v2generator.zip` es un `state_dict` nativo, no se deduce su formato de la extensión. Los tensores deben corresponder exactamente al generador original. Un archivo de seis canales se identifica como v1 y se rechaza. Los archivos TorchScript `colorizer.zip` también se rechazan antes de deserializar por la ruta de pesos.

El formato extendido actual es:

```text
format = "mangacolor.controlled.v1"
generator = state_dict de v2 sin claves palette_adapter
palette_adapter = state_dict exclusivo del adaptador
metadata = diccionario de procedencia y estado
epoch = entero
step = entero
optimizer = estado opcional del optimizador
```

La carga utiliza `torch.load(..., weights_only=True)` y validación estricta. No elimina prefijos arbitrarios ni rellena capas incompatibles silenciosamente. El programa no debe ofrecer una recarga insegura automática si la carga estricta falla.

### 16.2 Garantías y límites

| ID | Requisito | Estado |
| --- | --- | --- |
| CKP01 | Carga nativa y extendida con nombres y formas estrictos | I |
| CKP02 | Mantener generador y adaptador separados en serialización | I |
| CKP03 | Exportar un archivo cargable estrictamente por v2 original | I |
| CKP04 | Advertir que la exportación nativa omite la paleta global | I |
| CKP05 | Mantener salida base exacta al añadir adaptador de efecto cero bajo el mismo tensor | I |
| CKP06 | Versionar cambios de arquitectura y de contrato de preprocesado | P |
| CKP07 | Verificar procedencia y digest antes de seleccionar pesos en un proyecto | P |

La exportación nativa conserva las pistas locales, pero pierde la rama FiLM. Un generador afinado junto al adaptador puede comportarse de otra manera al retirarlo. La carga compatible demuestra integridad estructural; la calidad se evalúa por separado. No se promete que una v1 o un framework completo de borradores se convierta a v2 por cambiar nombres de claves.

## 17 Interfaces Python

### 17.1 Interfaces implementadas

```python
load_model(path, device="cpu", allow_untrained_palette=False)
# -> (ControlledColorizer en eval, metadata)

ColorizationPipeline(checkpoint, device="cpu")
ColorizationPipeline.colorize(
    image, hints=None, palette=None, size=576,
    palette_mode="hints", lock_hints=False, preserve_ink=False
)
# -> (PIL.Image RGB al tamano original, report)

prepare_dataset(color_dir, out_dir, bw_dir=None, val_fraction=.1, seed=42)
# -> Path del manifiesto

PaletteDataset(manifest, split="train", size=256, palette_size=8, seed=42)
PaletteDataset.set_epoch(epoch)

save_checkpoint(path, model, metadata, optimizer=None, epoch=0, step=0)
export_native(path, model)
```

### 17.2 Separación objetivo de predicción y renderizado

```python
predict_color_field(original, conditions, model, inference_config) -> ColorField
reconstruct(original, color_field, masks, reconstruction_config) -> RenderResult
export_result(render_result, destination, export_config) -> ExportRecord
```

`ColorField` contiene dimensiones canónicas, padding, RGB/Lab float32, transformaciones, clave de caché, hashes y procedencia. `RenderResult` contiene imagen, advertencias, estadísticas de gamut/protección y clave de reconstrucción. `ExportRecord` identifica ruta, digest y configuración final. Las tres funciones deben poder probarse sin servidor web.

## 18 API HTTP local

### 18.1 Contrato actual

El servidor se enlaza a `127.0.0.1`, puerto 7860 por defecto, con `HTTPServer` síncrono. No contiene autenticación ni trabajos de fondo. No se debe exponer a Internet sin otro diseño de servicio.

| Método y ruta | Entrada | Resultado |
| --- | --- | --- |
| GET `/` | Ninguna | HTML del editor |
| GET `/capabilities` | Ninguna | Metadatos del modelo |
| POST `/colorize` | JSON con imágenes PNG base64 | JSON con PNG base64 e informe |

Petición actual:

```json
{
  "image": "data:image/png;base64,<png RGB>",
  "hints": "data:image/png;base64,<png RGBA>",
  "palette": ["#285DAD", "#DBA98A"],
  "size": 576,
  "palette_mode": "hints",
  "lock_hints": false,
  "preserve_ink": true
}
```

`hints` es obligatorio en el HTTP actual, aunque la función Python acepta ausencia. Para no dar pistas se envía una capa transparente. El decodificador HTTP acepta PNG y rechaza más de 25 millones de píxeles por imagen. El cuerpo debe medir entre 1 y 40.000.000 bytes; base64 añade tamaño respecto al archivo binario.

La respuesta tiene `image` y `report`. El informe incluye modo de paleta, bloqueo, metadatos y dimensiones de inferencia; el modo nativo añade cantidad de muestras. Las rutas desconocidas devuelven 404; los errores manejados devuelven 400 con `error`. No existe todavía un catálogo estructurado de errores.

### 18.2 API objetivo versionada

Se propone `/api/v1` con creación y lectura de proyectos, actualización de perfiles y anotaciones, trabajos de predicción y exportaciones. Las imágenes grandes deben transferirse como binario/multipart o mediante referencias internas de activos ya importados, evitando duplicar cada original en cada petición base64.

```text
POST /api/v1/projects
POST /api/v1/projects/{id}/assets
PUT  /api/v1/projects/{id}/characters/{character_id}
PUT  /api/v1/projects/{id}/pages/{page_id}/annotations
POST /api/v1/predictions
GET  /api/v1/jobs/{job_id}
POST /api/v1/jobs/{job_id}/cancel
POST /api/v1/exports
```

Estos endpoints son propuestos. Una predicción devuelve 202 y `job_id`; una exportación reutiliza `prediction_id`. El contrato debe emplear revisiones o ETag para evitar que una respuesta tardía sobrescriba anotaciones recientes.

Los estados objetivo son `queued`, `preprocessing`, `predicting`, `reconstructing`, `saving`, `completed`, `cancelled` y `failed`. Una cancelación se atiende entre etapas; no se promete interrumpir instantáneamente un kernel CUDA. La cola por defecto tendrá un trabajador de inferencia por dispositivo para limitar memoria.

### 18.3 Errores objetivo

| Código | HTTP | Significado y recuperación |
| --- | --- | --- |
| `INVALID_IMAGE` | 422 | Archivo no decodificable; conservar proyecto |
| `MASK_SIZE_MISMATCH` | 422 | Corregir máscara o transformación |
| `MODEL_INCOMPATIBLE` | 422 | Seleccionar checkpoint compatible |
| `PALETTE_NOT_TRAINED` | 422 | Usar modo nativo o checkpoint entrenado |
| `LIMIT_EXCEEDED` | 413 | Reducir entrada o elegir ruta de importación adecuada |
| `REVISION_CONFLICT` | 409 | Refrescar y resolver revisiones |
| `OUTPUT_EXISTS` | 409 | Elegir destino; no sobrescribir implícitamente |
| `DEVICE_OUT_OF_MEMORY` | 503 | Liberar trabajo y proponer menor escala explícita |
| `INTERNAL_ERROR` | 500 | Registrar id de error sin descartar anotaciones |

No se cambiará silenciosamente de escala, precisión o dispositivo después de un OOM; hacerlo podría alterar color y debe producir una ejecución nueva registrada.

## 19 Interfaz de comandos

### 19.1 Comandos actuales

```powershell
.\.venv\Scripts\python.exe -m mangacolor inspect v2generator.zip
.\.venv\Scripts\python.exe -m mangacolor edit --checkpoint v2generator.zip
.\.venv\Scripts\python.exe -m mangacolor colorize pagina.png resultado.png --hints pistas.png
.\.venv\Scripts\python.exe -m mangacolor prepare fuentes\color data\manga
.\.venv\Scripts\python.exe -m mangacolor pseudo-label data\bw\manifest.jsonl data\pseudo
.\.venv\Scripts\python.exe -m mangacolor train data\manga\manifest.jsonl runs\experimento
.\.venv\Scripts\python.exe -m mangacolor export-v2 runs\experimento\last.pt runs\generator.zip
```

El argumento global `--threads`, por defecto 4, se coloca antes del subcomando.

| Grupo | Parámetros y valores actuales |
| --- | --- |
| Inferencia | `--checkpoint v2generator.zip`, `--device cpu`, `--size 576` |
| Control | `--hints`, `--palette` con HEX, `--palette-mode hints o learned`, `--lock-hints`, `--preserve-ink` |
| Preparación | `--bw-dir`, `--val-fraction 0.1`, `--seed 42` |
| Entrenamiento | `--size 256`, `--epochs 10`, `--batch-size 1`, `--lr 2e-5` |
| Entrenamiento adicional | `--scope decoder o adapter`, `--workers 0`, `--max-steps`, `--validation-batches`, `--resume` |
| Editor | `--port 7860`, checkpoint y dispositivo |

La CLI actual colorea un archivo por invocación. El soporte de carpetas, CBZ o CBR del proyecto v1 no se presume trasladado a esta reescritura.

### 19.2 Parámetros objetivo

Se añadirán `--inference-profile`, `--working-size`, `--output-size`, `--reconstruction-mode`, `--text-mask`, `--ink-mask`, `--project`, `--character-profiles` y `--reuse-prediction`. Estos nombres son propuestos y deben validarse al implementar la versión pública.

El antiguo `--size` conservará su significado o requerirá una deprecación explícita. No se reutilizará silenciosamente para tamaño final. Los ejemplos del producto deberán utilizar tamaños de trabajo y de exportación por separado.

## 20 Editor y experiencia de usuario

La base permite cargar una página, elegir color/grosor, pintar, rellenar, borrar, deshacer, definir paleta y descargar pistas o resultado. El relleno se basa en conectividad con un umbral de luminosidad de la imagen; un contorno abierto puede desbordarse y no equivale a segmentación semántica.

El editor objetivo mostrará original, pistas y resultado, con zoom sincronizado a coordenadas originales. El cambio de zoom no rasterizará de nuevo las anotaciones. Mantendrá capas separadas para pistas libres, regiones de perfil, protección y bloqueos exactos.

Se mostrarán de manera explícita el modo de paleta, si el checkpoint es experimental, si el resultado usa composición exacta y si existen conflictos con texto o tinta protegidos. Un estado de progreso no debe aparentar que la página está lista antes de guardar imagen e informe.

El control de resolución tendrá dos conceptos: escala de cálculo y tamaño de exportación. La acción de exportar a otro tamaño no pondrá el trabajo en estado `predicting` cuando el campo ya esté disponible. El usuario podrá ver qué predicción se reutiliza.

La accesibilidad objetivo incluye navegación de teclado para controles, etiquetas, foco visible, estados anunciados y contraste legible. El lienzo tendrá una alternativa de importación de máscaras para quien no pueda dibujar con puntero. La revisión visual y de accesibilidad completa está pendiente; las pruebas HTTP no la sustituyen.

## 21 Configuración objetivo y política de revisiones

Ejemplo propuesto, no cargado todavía por la CLI actual:

```json
{
  "schema": "mangacolor.render_config.v1",
  "inference": {
    "profile": "canonical_long_edge_v1",
    "working_size": 576,
    "device": "cpu",
    "precision": "float32",
    "palette_mode": "learned"
  },
  "reconstruction": {
    "algorithm": "guided_detail_v1",
    "color_space": "CIELAB_D65",
    "detail_strength": 1.0,
    "original_luminance_mix": 0.0,
    "gamut_mapping": "chroma_reduction",
    "protect_text": true,
    "protect_ink": true,
    "conflict_policy": "protected_pixels_first"
  },
  "export": {
    "size": "original",
    "format": "png",
    "overwrite": false
  }
}
```

Parámetros iniciales de reconstrucción propuestos: radio de guía de 4 píxeles canónicos, regularización de 0,001 sobre guía normalizada a `[0,1]`, sigma de detalle de 1 píxel canónico y 12 iteraciones de búsqueda binaria para reducción de cromaticidad fuera de gama. Los radios a resolución original se obtienen multiplicando por la relación de escalas, con redondeo mínimo a un píxel. Para un filtro guiado con doble promediado, el margen de tile será como mínimo dos veces el radio efectivo. Estos valores deben superar AT10 antes de convertirse en defaults de producto; no son parámetros presentes en 0.1.0.

Los radios, regularización, método de bordes y política de padding deberán formar parte de la versión final del algoritmo y del informe. No se introducirán defaults distintos en CLI y editor. Las revisiones de un perfil o máscara generan un nuevo hash y dejan las exportaciones previas reproducibles.

## 22 Seguridad operativa y privacidad

El producto local no necesita cuentas ni transmisión de imágenes. La configuración predeterminada mantiene loopback y no descarga pesos, datasets o modelos adicionales durante una colorización. Una descarga explícita futura deberá registrar origen, digest y versión.

Los límites se comprueban antes de reservar memoria excesiva. Se mantendrán las defensas de Pillow frente a imágenes descomprimidas enormes y se añadirán límites uniformes también a CLI y llamadas de servicio. Los archivos de proyecto se resuelven dentro de su directorio y no se aceptarán rutas que escapen al paquete durante importación.

Los checkpoints usan carga de pesos restringida; los campos canónicos se guardan sin objetos Python. Esto reduce superficies de deserialización, pero no convierte cualquier archivo de origen desconocido en seguro frente a vulnerabilidades del runtime. Las versiones publicadas deberán usar dependencias mantenidas y hashes de modelos conocidos.

Los logs no contendrán imágenes base64 ni datos privados de páginas. Se registrarán identificadores, configuración técnica, duraciones y errores. La limpieza de caché no afectará a originales ni a máscaras. No se incluyen operaciones de publicación ni servicios externos en este alcance.

El servidor objetivo rechazará orígenes ajenos cuando corresponda y limitará tamaño, tiempo y concurrencia. La versión actual no es una API endurecida para múltiples usuarios ni una autorización para exponer el puerto públicamente.

## 23 Dependencias y despliegue

### 23.1 Base declarada

| Dependencia | Restricción declarada |
| --- | --- |
| Python | 3.10 o superior |
| PyTorch | 2.6 o superior y menor que 3 |
| NumPy | 1.26 o superior y menor que 3 |
| Pillow | 10 o superior y menor que 13 |
| pytest opcional de desarrollo | 8 o superior y menor que 10 |

No se necesita torchvision para la copia vendorizada ni OpenCV para la preparación actual. El entorno de verificación inicial usa PyTorch 2.9.1 CPU. Las restricciones amplias del proyecto permiten instalar, pero no constituyen un lockfile reproducible ni una matriz de compatibilidad comprobada para todas las versiones.

### 23.2 Hardware y memoria

Se detectó anteriormente una RTX 4050 con aproximadamente 6 GB de VRAM. Esa observación no acredita rendimiento de entrenamiento CUDA: las verificaciones documentadas se hicieron en CPU. El perfil inicial de entrenamiento usa 256 y lote 1; `adapter` reduce parámetros entrenables, pero el gradiente debe recorrer partes del generador y su consumo debe medirse.

Para 1125 x 1600, una matriz float32 de tres canales ocupa aproximadamente 21,6 MB decimales. Una reconstrucción mantiene varias matrices, máscaras y temporales; no se debe estimar su memoria total como una sola imagen. El servicio objetivo establecerá una cuota de memoria y utilizará tiles con contexto para originales grandes sin cambiar la predicción global.

### 23.3 Instalación y operación

La ejecución desde la raíz usa `.venv` y `python -m mangacolor`. Una publicación instalable incluirá recursos HTML, versión de esquema y pruebas de importación desde un directorio distinto. Los pesos grandes se distribuyen por separado con procedencia verificada.

No se activa CUDA solo porque exista una GPU: debe existir una instalación de PyTorch compatible y `torch.cuda.is_available()` debe ser verdadero. Los errores se mostrarán antes del trabajo. Una futura elección automática de dispositivo se registrará; no deberá invalidar reproducibilidad sin avisar.

## 24 Evidencia de verificación existente

### 24.1 Resultados mecánicos

La base contiene 30 pruebas: 13 de datos, 9 de modelo, 7 de inferencia y pérdidas y 1 del editor HTTP. El 16 de noviembre de 2026 se repitieron inicialmente 29 sin red, con resultado 29 aprobadas en 10,81 s. Después de ampliar permisos se repitió la suite completa con el checkpoint real: **30 aprobadas en 15,36 s**, incluida la prueba HTTP. El log conservado es `docs/evidence/ssd-tests-20261116.txt`.

La primera ejecución restringida había fallado por permisos de reemplazo de archivos en temporales y por sockets no permitidos. La repetición utilizó temporales dentro del proyecto y los permisos ampliados, sin modificar el código para eludir pruebas. El inventario con hashes de la base examinada se conserva en `docs/ssd-baseline.json`; no existe un commit Git identificado en este directorio de trabajo.

Los controles cubren agrupación y duplicados, ausencia de etiquetas ficticias B/N, alineación geométrica, máscaras negras, semillas, protección de dataset existente, pesos pseudo, fuga train/val, carga real de v2, exportación estricta, efecto cero inicial, invariancia de paleta, gradientes, límites de pistas y respuesta HTTP con bloqueo exacto.

No existen todavía pruebas implementadas de conservación de texto mediante máscaras, caché canónica, consistencia de personajes, calidad artística entre obras o robustez a tamaños reales de inferencia. Deben figurar como pendientes, no como aprobadas.

### 24.2 Sensibilidad a pistas

`output/verification/model_verification.json` registra una comparación con el checkpoint real en un lienzo de 128 x 128. Un parche de 16 x 16 se cambia entre rojo y azul. La diferencia RGB absoluta media en escala `[0,1]` fue aproximadamente 0,521 dentro de la pista y 0,00468 fuera; globalmente 0,01275.

Esto acredita sensibilidad a la entrada cromática y un efecto localizado en ese ejemplo. No prueba identificación de pelo o piel, cobertura de un objeto entero ni ausencia general de fugas. Los tiempos CPU orientativos registrados con dos hilos son 0,279 s a 64 y 0,884 s a 128, medias de tres ejecuciones; no se extrapolan como SLA a 576.

### 24.3 Prueba de entrenamiento

`output/pipeline_check/verification.json` registra tres pasos sobre figuras geométricas sintéticas y dos actualizaciones con paleta. La exportación nativa cargó estrictamente. Cambiar paleta roja por azul tras el ajuste produjo una diferencia media de 0,0131 en RGB `[0,255]`, una variación muy pequeña.

La prueba acredita que existen gradientes, guardado, recarga y conexión entre paleta y salida. No acredita que el adaptador ya obedezca de forma útil ni que supere al modelo original. El error de pistas de validación de esa prueba, alrededor de 37,74 en `[0,255]`, pertenece a figuras y muestras concretas; no es una métrica de calidad de manga real.

## 25 Evaluación y criterios de aceptación

Todos los umbrales de calidad de esta sección son **criterios iniciales propuestos**. Deben fijarse junto con la versión del conjunto reservado antes de entrenar para evitar ajustar el criterio al resultado.

### 25.1 Protocolo reproducible

Se compararán al menos v2 original, base reescrita sin adaptador, modelo extendido entrenado, exportación nativa y reconstrucción objetivo. Cada comparación usará las mismas imágenes normalizadas, condiciones equivalentes y entorno registrado. Se evaluará sin pistas, con pistas pequeñas, con regiones y con paleta por separado.

Las páginas de test deben pertenecer a obras reservadas. Se informarán medianas, percentil 95 y casos fallidos por estilo y por tipo de anotación. Los intervalos o remuestreos estadísticos agruparán por obra o personaje; no tratarán píxeles vecinos como observaciones independientes.

Las regiones para medir color base excluirán tinta, texto y sombras extremas mediante máscaras de evaluación revisadas. La coincidencia con el perfil se medirá sobre cromaticidad y por partes, sin exigir RGB idéntico bajo iluminaciones distintas.

### 25.2 Métricas

**Error local bruto.** MAE RGB dentro de máscara, escalado a `[0,255]`, sobre la salida de red antes de cualquier bloqueo. Es indispensable para demostrar control aprendido. Se reporta además la versión compuesta para verificar el mecanismo exacto, sin confundir ambas.

**Cambio fuera de región.** Diferencia entre predicciones con dos pistas sobre la misma imagen, medida fuera de una dilatación fija y documentada de la máscara. El resultado cuantifica fugas o efectos globales y no tiene un límite universal idéntico para toda escena.

**Color por personaje.** Distancia CIEDE2000 respecto a una referencia de color y dispersión entre apariciones de la misma parte, sobre regiones de luminosidad comparable. Se reporta también error de a*/b* para separar luminosidad. La identidad correcta de las máscaras se evalúa por separado.

**Detalle y texto.** Error absoluto de píxeles protegidos; similitud estructural de luminancia fuera de las zonas recoloreadas; preservación de bordes; y CER de OCR sobre texto anotado. OCR es auxiliar: la garantía exacta depende de restauración de píxeles y el CER no sustituye revisión visual.

**Estabilidad de exportación.** Conteo de llamadas al modelo y hash del campo canónico al exportar el mismo original a varios tamaños. La conservación de la predicción es una propiedad exacta; las imágenes finales pueden diferir por remuestreo y gamut.

**Gamut y artefactos.** Fracción de píxeles con reducción cromática, saturación recortada, halos en borde y costuras de tiles. Se usarán parches sintéticos con límites conocidos y revisión de ejemplos reales.

### 25.3 Matriz de aceptación

| Prueba | Requisitos | Criterio de aceptación | Estado |
| --- | --- | --- | --- |
| AT01 Carga base | CKP01, CKP05 | Igualdad exacta con generador original para mismo tensor y CPU fijada | Cubierta |
| AT02 Exportación | CKP03, CKP04 | Carga estricta y ausencia de claves del adaptador | Cubierta |
| AT03 Máscara negra | LOC01 | Negro y ausencia producen contratos tensoriales distintos | Cubierta |
| AT04 Bloqueo actual | LOC03 | Error RGB cero en píxeles con alfa 255 a tamaño original | Cubierta |
| AT05 Paleta vacía y orden | PAL04 | Modulación vacía nula y diferencia máxima menor que 1e-5 por permutación | Cubierta en módulo |
| AT06 Separación de datos | DAT03, DAT06 | Cero grupos compartidos train/val/test; pseudo excluido de test | Parcial |
| AT07 Exportar varios tamaños | RES01, RES02 | Cero nuevas llamadas de red y hash de campo idéntico | Pendiente |
| AT08 Texto protegido | RES04 | Error RGB máximo cero dentro de máscara binaria al tamaño original | Pendiente |
| AT09 Tinta protegida | RES03 | Igualdad exacta en máscara explícita; comparar resto por bordes | Pendiente |
| AT10 Reconstrucción guiada | RES03, RES06 | Sin costuras sintéticas mayores de 1 nivel RGB; revisión de halos | Pendiente |
| AT11 Color de personaje | CON01, CON02, CON04 | Reducción de al menos 20 % de dispersión cromática frente a base en test | Pendiente |
| AT12 Obediencia aprendida | LOC03, TRN04 | Reducción de al menos 20 % del MAE de pistas bruto, sin hard lock | Pendiente |
| AT13 Sin regresión | TRN04, TRN10 | SSIM de luminancia no cae más de 0,01 y ΔE no empeora más de 5 % frente a base | Pendiente |
| AT14 Reinicio operativo | NF04, NF07, NF08 | Un fallo entre etapas no altera originales ni pierde anotaciones guardadas | Pendiente |
| AT15 Publicación de calidad | NF11 | Informe reservado completo, ejemplos y revisión humana | Pendiente |

AT11, AT12 y AT13 requieren fijar referencia, máscaras, agregación y dataset antes de medir; sus porcentajes son objetivos iniciales y no resultados de la base. Si una mejora de color empeora texto o detalles, no se considera aceptada por superar solo una media cromática.

### 25.4 Regresiones específicas de los issues

Para el issue 17 se usarán originales fijos y exportaciones de 576, 1125 y 2250 píxeles de lado mayor, sin volver a inferir. Se probará por separado inferencia a distintas escalas para mostrar su sensibilidad real. No se mezclarán ambos experimentos en una sola afirmación de estabilidad.

Para el issue 39 se medirán varias apariciones por personaje y parte en al menos tres viñetas cuando existan, con variación de sombras y encuadres. Se comparará paleta global sola, perfil con máscaras y, posteriormente, correspondencia automática. El error de asignar la identidad o la parte incorrecta cuenta como fallo aunque el color elegido sea correcto para otro personaje.

## 26 Observabilidad y recuperación

El informe objetivo de cada resultado contendrá versión de esquema, página, checkpoint, hash de condiciones, perfil canónico, `prediction_id`, `render_id`, dimensiones, parámetros, revisiones de perfiles, protecciones, advertencias y tiempos por etapa. Se registrará si hubo reutilización de caché, composición exacta o reducción de gama.

Las métricas operativas incluyen duración y memoria de pico por etapa, tasa de caché, fallos de entrada, OOM y cancelaciones. Las métricas de calidad no se calculan como verdad de referencia cuando no hay objetivo real; en ese caso solo se registran verificaciones de integridad y observaciones revisadas.

Una escritura usa un nombre temporal en el mismo volumen, sincroniza cuando sea necesario y publica mediante reemplazo. Una exportación incompleta no aparece como finalizada. El sistema conserva el estado anterior válido y ofrece reintentar desde la última etapa con artefacto íntegro.

El rollback de un modelo cambia la referencia activa a un checkpoint previo sin modificar sus archivos. Los proyectos mantienen su versión de configuración y pueden reconstruir con el recorrido anterior. Las migraciones de esquemas son explícitas y reversibles mediante copia de respaldo.

## 27 Plan de implementación y entregables

### 27.1 Hito A Base trazable

Consolidar contratos actuales, lockfile y registro de entorno; unificar errores y límites; guardar configuración y hashes completos; mantener la suite actual. Salida: aplicación base reproducible y manifiestos auditables. No requiere declarar mejora artística.

### 27.2 Hito B Resolución independiente

Separar `predict_color_field`, `reconstruct` y `export_result`; implementar caché canónica, ampliación guiada, manejo de luminosidad/gamut y protecciones. Actualizar editor y CLI para mostrar ambos tamaños. Salida: AT07, AT08, AT09 y pruebas sintéticas AT10 aprobadas, más revisión visual real.

### 27.3 Hito C Perfiles por personaje

Implementar proyectos, perfiles versionados y vínculos manuales a regiones. Añadir prioridades y detección de conflictos. Salida: un capítulo puede reutilizar una ficha cromática sin depender de identificación automática. Evaluar AT11 sobre máscaras humanas antes de añadir detectores.

### 27.4 Hito D Datos y modelo validado

Construir colección con procedencia y test separado por obra; revisar parejas y pseudoetiquetas; entrenar alcances y pérdidas mediante ablaciones. Salida: checkpoints con informe reproducible, AT12, AT13 y AT15 evaluadas, y exportación v2 medida por separado.

### 27.5 Hito E Automatización y lotes

Añadir cola persistente, cancelación, importación por carpetas y archivos de capítulo, y propuestas automáticas de segmentación/correspondencia. Salida: procesamiento reanudable que mantiene las decisiones humanas. No condiciona la primera solución de resolución ni el control manual.

No se asignan fechas ni costes de GPU sin conocer dataset, volumen, disponibilidad y resultados. Cada hito se cierra por evidencias de aceptación, no por completar una lista de archivos.

## 28 Riesgos y decisiones de diseño

| Riesgo | Efecto | Respuesta y evidencia requerida |
| --- | --- | --- |
| Paleta sin posición | Colores correctos en objetos incorrectos | Perfiles y máscaras; evaluar identidad y parte |
| Adaptador con poca capacidad | Obediencia global débil | Medir y ampliar arquitectura solo con ablación |
| Blanco original y color saturado | Desaturación o salida de gama | Iluminación predicha y mapeo cromático |
| Color a baja resolución | Sangrado en detalles pequeños | Guía de bordes, regiones y evaluación local |
| Tramas y ruido | Halos o textura cromática artificial | Filtrado versionado y protección de detalle |
| Parejas mal alineadas | Aprendizaje borroso o contradictorio | Revisión y transformación registrada |
| Pseudoetiquetas erróneas | Reproducción de errores del profesor | Procedencia, revisión y mezcla controlada |
| Fuga entre obras | Métricas optimistas | Grupos congelados y detección de copias próximas |
| Cambio de preprocesado | Pérdida de reproducción de v2 | Perfiles versionados y comparación de tensor |
| Exportar sin adaptador | Regresión del generador afinado | Evaluación nativa separada |
| Memoria limitada | Fallos a alta resolución | Predicción canónica, tiles y cola por dispositivo |
| Detección de texto incompleta | Texto coloreado o alterado | Máscara humana como primera garantía |

**ADR01 Conservar la base v2.** Permite reutilizar pesos y pistas existentes. Su compatibilidad se demuestra con carga estricta y salidas bajo el mismo tensor.

**ADR02 Añadir paleta como conjunto.** DeepSets evita asociar significado al orden de los colores. El coste es que no expresa roles semánticos; esos roles pertenecen a perfiles regionales.

**ADR03 Separar predicción y reconstrucción.** Permite estabilidad de decisión al exportar sin exigir invariancia de la red a toda escala.

**ADR04 Preservar detalle y proteger píxeles explícitos.** La descomposición de luminosidad conserva textura de forma aproximada; las máscaras proporcionan garantías exactas en las zonas elegidas.

**ADR05 Empezar la consistencia con vínculo humano.** Evita condicionar el flujo útil a disponer de un reconocedor automático de personajes fiable.

**ADR06 Usar supervisión reproducible.** El entrenamiento parte de datos identificados y pérdidas actuales verificables. Los mecanismos de los artículos no presentes se añaden como experimentos, sin atribuir sus resultados al modelo disponible.

## 29 Trazabilidad de requisitos y módulos

| Grupo de requisitos | Módulos principales | Evidencia de aceptación |
| --- | --- | --- |
| IN01 a IN04 | Entrada, editor y pipeline | Formatos, EXIF, alfa, límites y coordenadas |
| LOC01 a LOC05 | Editor, anotaciones y pipeline | AT03, AT04, AT12 y proyecto reabierto |
| PAL01 a PAL05 | Parseo, modelo y checkpoint | AT05 y tests de rechazo |
| RES01 a RES06 | Campo canónico, caché y reconstrucción | AT07 a AT10 |
| CON01 a CON05 | Proyectos, perfiles y regiones | AT11 y revisión de correspondencias |
| DAT01 a DAT10 | Preparación, dataset y manifiesto | AT06 y auditoría de fuentes |
| TRN01 a TRN10 | Entrenamiento y evaluación | AT12, AT13 y AT15 |
| CKP01 a CKP07 | Checkpoints y exportación | AT01 y AT02 |
| OUT01 a OUT03 | Exportación y proyectos | Integridad, no sobrescritura e informe |
| NF01 a NF14 | Operación transversal | AT14, instalación y revisión de versión |

Los cambios futuros deben mencionar al menos un requisito y su prueba o justificación de diseño. Un requisito parcial no se marca como cerrado por probar solo su parte ya implementada.

## 30 Fuentes y referencias

### 30.1 Código y evidencias locales

- [R0] `README.md`, `pyproject.toml`, `mangacolor/`, `tests/` y `scripts/verify_pipeline.py`. Base de comportamiento implementado de la versión 0.1.0.
- [E1] `output/verification/model_verification.json` y comparación de pistas en el mismo directorio. Sensibilidad local y microbenchmarks CPU.
- [E2] `output/pipeline_check/verification.json` y `training/metrics.jsonl`. Prueba mecánica de entrenamiento.
- [E3] `docs/evidence/ssd-tests-20261116.txt` y `docs/ssd-baseline.json`. Resultado completo e inventario SHA256 de la base examinada. Las ejecuciones intermedias permanecen en `tmp/ssd_tests_20261116.log` y `tmp/ssd_full_tests_20261116.log`.
- [E4] `docs/RESEARCH.md`. Revisión de relación entre artículos, código público y decisiones de esta reescritura.

### 30.2 Fuentes primarias externas

- [R1] Issue 39 sobre consistencia de color y parámetros. https://github.com/qweasdd/manga-colorization-v2/issues/39
- [R2] Issue 17 sobre cambios de color al variar tamaño. https://github.com/qweasdd/manga-colorization-v2/issues/17
- [R3] Código original v2. https://github.com/qweasdd/manga-colorization-v2
- [R4] Golyadkin y Makarov. Semi automatic Manga Colorization Using Conditional Adversarial Networks, 2021. PDF local `2021-golyadkin.pdf`.
- [R5] Golyadkin y Makarov. Robust Manga Page Colorization via Coloring Latent Space, IEEE Access, 2023. PDF local con ese título. El framework completo incluye mecanismos que no están en el generador público utilizado aquí.
- [R6] Golyadkin, Plevokas y Makarov. Closing the Domain Gap in Manga Colorization via Aligned Paired Dataset, WACV 2025. https://openaccess.thecvf.com/content/WACV2025/html/Golyadkin_Closing_the_Domain_Gap_in_Manga_Colorization_via_Aligned_Paired_WACV_2025_paper.html
- [R7] He, Sun y Tang. Guided Image Filtering, ECCV 2010 y extensión TPAMI 2013. Explica ampliación conjunta guiada y su aplicación a canales cromáticos. https://people.csail.mit.edu/kaiming/eccv10/index.html
- [R8] Golyadkin, Saraev y Makarov. Advancing Sequential Manga Colorization for AR Through Data Synthesis, IEEE Access, 2025. PDF local; aborda fotografías y síntesis para AR, no una garantía de colores verdaderos desde B/N.

Las referencias documentan métodos y problemas. No constituyen resultados de esta implementación. Las copias locales originales no incluyen una licencia visible que permita asignarles automáticamente una licencia nueva; código, pesos y datasets requieren revisar sus propias condiciones antes de redistribuir o usar comercialmente. Este SSD no es una evaluación jurídica ni atribuye permisos que no constan.

SensorSCAN trata diagnóstico de procesos químicos y queda fuera de la arquitectura de colorización. Otros archivos del directorio sin relación con manga no son fuentes técnicas de este documento.

## 31 Glosario

| Término | Definición |
| --- | --- |
| SSD | Documento de especificación del diseño de software |
| Pista o hint | Indicación de color localizada que recibe la red |
| Máscara | Cobertura espacial independiente del valor RGB |
| Paleta global | Conjunto de colores sin ubicación ni roles semánticos |
| Perfil de personaje | Colores base por parte y referencias de una identidad |
| Campo canónico | Predicción de color conservada a escala de trabajo fija |
| Escala de inferencia | Tamaño del tensor que procesa la red |
| Tamaño de exportación | Dimensiones del archivo final, separadas de la inferencia |
| FiLM | Modulación de características por escala y desplazamiento |
| DeepSets | Codificación de un conjunto mediante funciones compartidas y agregación |
| Pseudoetiqueta | Objetivo propuesto por un modelo, sin condición de verdad de referencia |
| Gamut | Colores representables en el espacio y dispositivo de salida |
| Tinta protegida | Píxeles del original que se restauran por máscara explícita |
| Reanudación operativa | Recuperación que continúa entrenamiento sin reproducir exactamente el paso interrumpido |
| Carga estricta | Comprobación de todas las claves y formas esperadas de pesos |

## 32 Condiciones para considerar completado el producto

La primera versión con resolución estable estará completada cuando la predicción y la exportación estén separadas, las protecciones suministradas conserven sus píxeles bajo la política declarada, las salidas reutilicen el mismo campo y se haya revisado visualmente una colección real diversa. El control local debe medirse tanto antes como después de la composición.

La versión con consistencia por personaje estará completada cuando un proyecto pueda guardar perfiles, vincular partes en varias viñetas, regenerar de forma trazable y superar su protocolo reservado. La disponibilidad de un detector automático no sustituye la comprobación del resultado.

Un modelo mejorado se considerará validado únicamente después de entrenar con procedencia conocida y publicar la comparación con v2 en test reservado. Los checkpoints experimentales, las pruebas de gradiente y la compatibilidad estructural siguen siendo entregables útiles, con un alcance distinto de esa validación.
