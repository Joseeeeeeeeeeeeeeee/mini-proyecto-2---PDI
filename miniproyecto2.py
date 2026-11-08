import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from skimage.feature import graycomatrix, local_binary_pattern
from skimage.feature import graycoprops
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

# Rutas del proyecto
ruta_base = Path(__file__).resolve().parent
dataset = ruta_base / "Dataset"
ruta_rois = ruta_base / "ROIs"


#=======================================================================
#                                                                       #
#                         ACTIVIDAD 1 - ITEM 1.1                        #
#                ADQUISICION Y CONSTRUCCION DEL DATASET                 #
#                                                                       #
#=======================================================================

#-----------------------------------------------------------------------
#------------------ Lectura de imagenes --------------------------------
#-----------------------------------------------------------------------

imagenes = sorted(glob.glob(str(dataset / "Tela *" / "*")))

for i in imagenes:

    if Path(i).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    imagen = cv.imdecode(
        np.fromfile(i, dtype=np.uint8),
        cv.IMREAD_COLOR
    )

    if imagen is None:
        print(f"Error al cargar la imagen {i}")

    else:
        imagen_RGB = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)

        clase = Path(i).parent.name

        print(f"{clase}: {Path(i).name}")


#=======================================================================
#                                                                       #
#                         ACTIVIDAD 1 - ITEM 1.2                        #
#               PREPROCESAMIENTO Y CARACTERISTICAS VISUALES             #
#                                                                       #
#=======================================================================


#-----------------------------------------------------------------------
#------------------ 1.2.1 Preparacion de las imagenes ------------------
#-----------------------------------------------------------------------


#------------------ Extraccion de ROI ----------------------------------

def extraer_rois(nombre_archivo):

    imagen = cv.imdecode(
        np.fromfile(nombre_archivo, dtype=np.uint8),
        cv.IMREAD_COLOR
    )

    if imagen is None:
        raise FileNotFoundError(nombre_archivo)

    imagen_RGB = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)

    alto, ancho = imagen_RGB.shape[:2]
    lado = min(alto, ancho) // 4

    centros = [
        (ancho // 4, alto // 4),
        (3 * ancho // 4, alto // 4),
        (ancho // 4, 3 * alto // 4),
        (3 * ancho // 4, 3 * alto // 4)
    ]

    rois = []

    for cx, cy in centros:

        x_0 = cx - lado // 2
        y_0 = cy - lado // 2

        roi = imagen_RGB[
            y_0:y_0 + lado,
            x_0:x_0 + lado
        ]

        rois.append(roi.copy())

    return rois


#------------------ Visualizacion de ROI -------------------------------

def mostrar_rois(nombre_archivo):

    imagen = cv.imdecode(
        np.fromfile(nombre_archivo, dtype=np.uint8),
        cv.IMREAD_COLOR
    )

    if imagen is None:
        raise FileNotFoundError(nombre_archivo)

    imagen_RGB = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)

    alto, ancho = imagen_RGB.shape[:2]
    lado = min(alto, ancho) // 4

    centros = [
        (ancho // 4, alto // 4),
        (3 * ancho // 4, alto // 4),
        (ancho // 4, 3 * alto // 4),
        (3 * ancho // 4, 3 * alto // 4)
    ]

    imagen_marcada = imagen_RGB.copy()

    rois = []

    fig, ax = plt.subplots(1, 5, figsize=(18, 4))

    for n, (cx, cy) in enumerate(centros):

        x_0 = cx - lado // 2
        y_0 = cy - lado // 2

        roi = imagen_RGB[
            y_0:y_0 + lado,
            x_0:x_0 + lado
        ]

        rois.append(roi.copy())

        cv.rectangle(
            imagen_marcada,
            (x_0, y_0),
            (x_0 + lado, y_0 + lado),
            (255, 0, 0),
            5
        )

        ax[n + 1].imshow(roi)
        ax[n + 1].set_title(f"ROI {n + 1}")
        ax[n + 1].axis("off")

    ax[0].imshow(imagen_marcada)
    ax[0].set_title("Imagen original")
    ax[0].axis("off")

    plt.tight_layout()
    plt.show()

    return rois


#------------------ Verificacion del tamaño de ROI ---------------------

#tamanos = set()

#for clase in sorted(ruta_rois.iterdir()):

#    if clase.is_dir():

#        for archivo in clase.glob("*.png"):

#            imagen = cv.imdecode(
#                np.fromfile(archivo, dtype=np.uint8),
#                cv.IMREAD_COLOR
#            )

#            if imagen is not None:
#                alto, ancho = imagen.shape[:2]
#                tamanos.add((ancho, alto))

#print(f"Tamaños encontrados: {sorted(tamanos)}")


#------------------ Verificacion de ROI --------------------------------

mostrar_verificacion = False

if mostrar_verificacion:

    for clase in sorted(dataset.iterdir()):

        if clase.is_dir():

            archivos = sorted([
                archivo for archivo in clase.iterdir()
                if archivo.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ])

            if archivos:

                print(f"Mostrando ROI de {clase.name}")

                mostrar_rois(str(archivos[0]))


#------------------ Almacenamiento de ROI ------------------------------

guardar_rois = False

if guardar_rois:

    total_rois = 0

    for clase in sorted(dataset.iterdir()):

        if clase.is_dir():

            carpeta_salida = ruta_rois / clase.name
            carpeta_salida.mkdir(parents=True, exist_ok=True)

            archivos = sorted([
                archivo for archivo in clase.iterdir()
                if archivo.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ])

            for archivo in archivos:

                rois = extraer_rois(str(archivo))

                for n, roi in enumerate(rois):

                    nombre = f"{archivo.stem}_ROI_{n+1}.png"
                    destino = carpeta_salida / nombre

                    roi_BGR = cv.cvtColor(roi, cv.COLOR_RGB2BGR)

                    correcto, datos = cv.imencode(".png", roi_BGR)

                    if not correcto:
                        raise RuntimeError(
                            f"Error al codificar {nombre}"
                        )

                    datos.tofile(str(destino))

                    total_rois += 1

            print(f"{clase.name}: {len(archivos)*4} ROI")

    print(f"\nTotal de ROI guardadas: {total_rois}")


#------------------ Comparacion Original, R, G, B y grises -------------

mostrar_comparacion = False

if mostrar_comparacion:

    for inicio in range(0, 8, 4):

        fig, ax = plt.subplots(
            4, 5,
            figsize=(16, 12),
            layout="constrained"
        )

        fig.suptitle(
            "Comparacion: Original, canal R, canal G, canal B y escala de grises",
            fontsize=16
        )

        for n in range(inicio, inicio + 4):

            clase = f"Tela {n+1}"

            archivos = sorted((ruta_rois / clase).glob("*.png"))

            if not archivos:
                raise FileNotFoundError(f"No hay ROI en {clase}")

            imagen = cv.imdecode(
                np.fromfile(archivos[0], dtype=np.uint8),
                cv.IMREAD_COLOR
            )

            if imagen is None:
                raise ValueError(f"No se pudo leer {archivos[0]}")

            imagen_RGB = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)
            imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)

            canal_R, canal_G, canal_B = cv.split(imagen_RGB)

            fila = n - inicio

            # Columna 1: Original
            ax[fila, 0].imshow(imagen_RGB)
            ax[fila, 0].set_title(f"{clase} - Original", fontsize=10)
            ax[fila, 0].axis("off")

            # Columna 2: Canal R
            ax[fila, 1].imshow(canal_R, cmap="gray", vmin=0, vmax=255)
            ax[fila, 1].set_title(f"{clase} - Canal R", fontsize=10)
            ax[fila, 1].axis("off")

            # Columna 3: Canal G
            ax[fila, 2].imshow(canal_G, cmap="gray", vmin=0, vmax=255)
            ax[fila, 2].set_title(f"{clase} - Canal G", fontsize=10)
            ax[fila, 2].axis("off")

            # Columna 4: Canal B
            ax[fila, 3].imshow(canal_B, cmap="gray", vmin=0, vmax=255)
            ax[fila, 3].set_title(f"{clase} - Canal B", fontsize=10)
            ax[fila, 3].axis("off")

            # Columna 5: Escala de grises
            ax[fila, 4].imshow(imagen_gris, cmap="gray", vmin=0, vmax=255)
            ax[fila, 4].set_title(f"{clase} - Grises", fontsize=10)
            ax[fila, 4].axis("off")

        plt.show()


#------------------ Comparacion de tecnicas de realce y filtrado --------

mostrar_procesamientos = False #Ninguna "mejora"

if mostrar_procesamientos:

    # Todos los resultados se obtienen directamente de la misma ROI gris.
    # Los valores elegidos son parametros iniciales para una comparacion.

    def normalizar_respuesta(respuesta):
        respuesta = np.abs(respuesta).astype(np.float32)
        return cv.normalize(
            respuesta, None, 0, 255, cv.NORM_MINMAX
        ).astype(np.uint8)

    def magnitud_gradiente(imagen, kernel_x, kernel_y):
        gx = cv.filter2D(imagen, cv.CV_32F, kernel_x)
        gy = cv.filter2D(imagen, cv.CV_32F, kernel_y)
        return normalizar_respuesta(cv.magnitude(gx, gy))

    def aplicar_tecnicas(imagen_gris):
        gris = imagen_gris.astype(np.float32)

        # Transformaciones puntuales (Capitulo 2, Parte I)
        brillo = np.clip(gris + 30, 0, 255).astype(np.uint8)
        contraste = np.clip(1.35 * (gris - 128) + 128, 0, 255).astype(np.uint8)
        negativo = cv.bitwise_not(imagen_gris)
        logaritmica = np.clip(
            255 * np.log1p(gris) / np.log(256), 0, 255
        ).astype(np.uint8)
        gamma_06 = np.clip(255 * (gris / 255) ** 0.6, 0, 255).astype(np.uint8)
        gamma_16 = np.clip(255 * (gris / 255) ** 1.6, 0, 255).astype(np.uint8)
        tramos = np.interp(
            imagen_gris, [0, 60, 170, 255], [0, 40, 220, 255]
        ).astype(np.uint8)

        # Histograma (Capitulo 2, Parte II)
        estiramiento = cv.normalize(
            imagen_gris, None, 0, 255, cv.NORM_MINMAX
        )
        ecualizacion = cv.equalizeHist(imagen_gris)

        # clipLimit=0: ecualizacion local sin recorte del histograma (AHE).
        ahe = cv.createCLAHE(
            clipLimit=0.0, tileGridSize=(8, 8)
        ).apply(imagen_gris)
        clahe = cv.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8)
        ).apply(imagen_gris)

        # Filtros espaciales (Capitulo 2, Parte V)
        promedio = cv.blur(imagen_gris, (3, 3))
        gaussiano = cv.GaussianBlur(imagen_gris, (3, 3), 0)
        mediana = cv.medianBlur(imagen_gris, 3)
        kernel_3 = np.ones((3, 3), dtype=np.uint8)
        minimo = cv.erode(imagen_gris, kernel_3)
        maximo = cv.dilate(imagen_gris, kernel_3)

        # Operadores de gradiente (Capitulo 2, Parte V)
        roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
        roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        prewitt_x = np.array([
            [-1, 0, 1], [-1, 0, 1], [-1, 0, 1]
        ], dtype=np.float32)
        prewitt_y = np.array([
            [-1, -1, -1], [0, 0, 0], [1, 1, 1]
        ], dtype=np.float32)

        roberts = magnitud_gradiente(imagen_gris, roberts_x, roberts_y)
        prewitt = magnitud_gradiente(imagen_gris, prewitt_x, prewitt_y)
        sobel = normalizar_respuesta(cv.magnitude(
            cv.Sobel(imagen_gris, cv.CV_32F, 1, 0, ksize=3),
            cv.Sobel(imagen_gris, cv.CV_32F, 0, 1, ksize=3)
        ))
        scharr = normalizar_respuesta(cv.magnitude(
            cv.Scharr(imagen_gris, cv.CV_32F, 1, 0),
            cv.Scharr(imagen_gris, cv.CV_32F, 0, 1)
        ))

        laplaciano = cv.Laplacian(imagen_gris, cv.CV_32F, ksize=1)
        mapa_laplaciano = normalizar_respuesta(laplaciano)
        realce_laplaciano = np.clip(
            gris - laplaciano, 0, 255
        ).astype(np.uint8)

        # LoG y Canny: mapas de bordes para evaluar, no realce definitivo.
        suavizada = cv.GaussianBlur(imagen_gris, (3, 3), 0)
        respuesta_log = cv.Laplacian(suavizada, cv.CV_32F, ksize=1)
        log_magnitud = normalizar_respuesta(respuesta_log)
        canny = cv.Canny(imagen_gris, 60, 120)

        return {
            'Grises': imagen_gris,
            'Brillo +30': brillo,
            'Contraste 1.35': contraste,
            'Negativo': negativo,
            'Logaritmica': logaritmica,
            'Gamma 0.6': gamma_06,
            'Gamma 1.6': gamma_16,
            'Por tramos': tramos,
            'Estiramiento': estiramiento,
            'Ecualizacion': ecualizacion,
            'AHE': ahe,
            'CLAHE': clahe,
            'Promedio 3x3': promedio,
            'Gaussiano 3x3': gaussiano,
            'Mediana 3x3': mediana,
            'Minimo 3x3': minimo,
            'Maximo 3x3': maximo,
            'Roberts': roberts,
            'Prewitt': prewitt,
            'Sobel': sobel,
            'Scharr': scharr,
            'Laplaciano': mapa_laplaciano,
            'Realce Laplaciano': realce_laplaciano,
            'LoG (magnitud)': log_magnitud,
            'Canny': canny
        }

    # Cada familia se compara por separado para que las figuras sean legibles.
    grupos_tecnicas = {
        'Transformaciones de intensidad I': [
            'Grises', 'Brillo +30', 'Contraste 1.35', 'Negativo', 'Logaritmica'
        ],
        'Transformaciones de intensidad II': [
            'Grises', 'Gamma 0.6', 'Gamma 1.6', 'Por tramos'
        ],
        'Procesamiento de histogramas': [
            'Grises', 'Estiramiento', 'Ecualizacion', 'AHE', 'CLAHE'
        ],
        'Filtros espaciales de suavizado': [
            'Grises', 'Promedio 3x3', 'Gaussiano 3x3',
            'Mediana 3x3', 'Minimo 3x3', 'Maximo 3x3'
        ],
        'Operadores de gradiente': [
            'Grises', 'Roberts', 'Prewitt', 'Sobel', 'Scharr'
        ],
        'Laplaciano y detectores de bordes': [
            'Grises', 'Laplaciano', 'Realce Laplaciano',
            'LoG (magnitud)', 'Canny'
        ]
    }

    # La primera ROI de cada clase es la misma de la comparacion R, G, B.
    muestras = []

    for n in range(1, 9):
        clase = f'Tela {n}'
        archivos = sorted((ruta_rois / clase).glob('*.png'))
        if not archivos:
            raise FileNotFoundError(f'No hay ROI en {clase}')

        imagen = cv.imdecode(
            np.fromfile(archivos[0], dtype=np.uint8), cv.IMREAD_COLOR
        )
        if imagen is None:
            raise ValueError(f'No se pudo leer {archivos[0]}')

        imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
        muestras.append((clase, aplicar_tecnicas(imagen_gris)))

    for nombre_grupo, columnas in grupos_tecnicas.items():
        for inicio in range(0, len(muestras), 4):
            grupo = muestras[inicio:inicio + 4]
            fig, ax = plt.subplots(
                len(grupo), len(columnas),
                figsize=(3.3 * len(columnas), 3 * len(grupo)),
                squeeze=False, layout='constrained'
            )
            fig.suptitle(nombre_grupo, fontsize=15)

            for fila, (clase, resultados) in enumerate(grupo):
                for columna, tecnica in enumerate(columnas):
                    ax[fila, columna].imshow(
                        resultados[tecnica], cmap='gray', vmin=0, vmax=255
                    )
                    ax[fila, columna].set_title(
                        f'{clase} - {tecnica}', fontsize=9
                    )
                    ax[fila, columna].axis('off')

            plt.show()


#-----------------------------------------------------------------------
#---------- 1.2.2 Identificacion y extraccion de -----------------------
#------------------ caracteristicas visuales ---------------------------
#-----------------------------------------------------------------------


mostrar_texturas = True

if mostrar_texturas:

    # Configuraciones iniciales para comparar las cuatro tecnicas.

    angulos = [0, 45, 90, 135]

    vectores_laws = {
        "L5": np.array([1, 4, 6, 4, 1], dtype=np.float32),
        "E5": np.array([-1, -2, 0, 2, 1], dtype=np.float32),
        "S5": np.array([-1, 0, 2, 0, -1], dtype=np.float32),
        "W5": np.array([-1, 2, 0, -2, 1], dtype=np.float32),
        "R5": np.array([1, -4, 6, -4, 1], dtype=np.float32)
    }

    mascaras_laws = [
        "L5E5", "E5L5", "L5S5",
        "S5S5", "W5W5", "R5R5"
    ]


    #------------------ Extraccion de textura GLCM ---------------------

    def aplicar_glcm(imagen_gris):

        # Reducimos de 256 a 16 niveles de intensidad.
        imagen_16 = imagen_gris // 16

        matriz = graycomatrix(
            imagen_16,
            distances=[1],
            angles=np.deg2rad(angulos),
            levels=16,
            symmetric=True,
            normed=True
        )

        resultados = {}

        for n, angulo in enumerate(angulos):
            resultados[f"GLCM {angulo}"] = matriz[:, :, 0, n]

        return resultados


    #------------------ Extraccion de textura LBP ----------------------

    def aplicar_lbp(imagen_gris):

        lbp = local_binary_pattern(
            imagen_gris,
            P=8,
            R=1,
            method="uniform"
        )

        histograma = np.bincount(
            lbp.astype(np.int32).ravel(),
            minlength=10
        )[:10].astype(np.float64)

        histograma = histograma / histograma.sum()

        return lbp, histograma


    #------------------ Extraccion de textura Gabor --------------------

    def aplicar_gabor(imagen_gris):

        resultados = {}

        for angulo in angulos:

            kernel = cv.getGaborKernel(
                (21, 21),
                sigma=5,
                theta=np.deg2rad(angulo),
                lambd=8,
                gamma=0.5,
                psi=0,
                ktype=cv.CV_32F
            )

            # Eliminamos la componente media y normalizamos el kernel.
            kernel = kernel - kernel.mean()
            kernel = kernel / (np.sum(np.abs(kernel)) + 1e-8)

            respuesta = cv.filter2D(
                imagen_gris.astype(np.float32),
                cv.CV_32F,
                kernel,
                borderType=cv.BORDER_REFLECT
            )

            # Mapa de energia local de la respuesta.
            energia = cv.GaussianBlur(
                np.abs(respuesta),
                (7, 7),
                0
            )

            resultados[f"Gabor {angulo}"] = energia

        return resultados


    #------------------ Extraccion de textura Laws ---------------------

    def aplicar_laws(imagen_gris):

        resultados = {}

        imagen_float = imagen_gris.astype(np.float32)

        # Eliminacion de variaciones suaves de iluminacion.
        fondo = cv.GaussianBlur(
            imagen_float,
            (15, 15),
            0
        )

        imagen_centrada = imagen_float - fondo

        for mascara in mascaras_laws:

            a = mascara[:2]
            b = mascara[2:]

            kernel = np.outer(
                vectores_laws[a],
                vectores_laws[b]
            ).astype(np.float32)

            kernel = kernel / (np.sum(np.abs(kernel)) + 1e-8)

            respuesta = cv.filter2D(
                imagen_centrada,
                cv.CV_32F,
                kernel,
                borderType=cv.BORDER_REFLECT
            )

            # Mapa de energia local de Laws.
            energia = cv.GaussianBlur(
                np.abs(respuesta),
                (9, 9),
                0
            )

            resultados[f"Laws {mascara}"] = energia

        return resultados


    #------------------ Aplicacion sobre las ocho telas ----------------

    muestras = []

    for n in range(1, 9):

        clase = f"Tela {n}"

        archivos = sorted(
            (ruta_rois / clase).glob("*.png")
        )

        if not archivos:
            raise FileNotFoundError(f"No hay ROI en {clase}")

        imagen = cv.imdecode(
            np.fromfile(archivos[0], dtype=np.uint8),
            cv.IMREAD_COLOR
        )

        if imagen is None:
            raise ValueError(f"No se pudo leer {archivos[0]}")

        imagen_gris = cv.cvtColor(
            imagen,
            cv.COLOR_BGR2GRAY
        )

        # Aplicamos las cuatro tecnicas de forma independiente.
        resultados_glcm = aplicar_glcm(imagen_gris)
        resultado_lbp, histograma_lbp = aplicar_lbp(imagen_gris)
        resultados_gabor = aplicar_gabor(imagen_gris)
        resultados_laws = aplicar_laws(imagen_gris)

        datos = {
            "clase": clase,
            "Grises": imagen_gris,
            "LBP": resultado_lbp,
            "Histograma LBP": histograma_lbp
        }

        datos.update(resultados_glcm)
        datos.update(resultados_gabor)
        datos.update(resultados_laws)

        muestras.append(datos)

        print(f"{clase}: textura extraida de {archivos[0].name}")


    #------------------ Escalas comunes de visualizacion ---------------

    # Usamos una escala compartida para cada familia de mapas,
    # evitando normalizar cada ROI de manera independiente.

    vmax_glcm = max(
        np.max(datos[f"GLCM {angulo}"])
        for datos in muestras
        for angulo in angulos
    )

    vmax_gabor = max(
        np.percentile(datos[f"Gabor {angulo}"], 99)
        for datos in muestras
        for angulo in angulos
    )

    vmax_laws = max(
        np.percentile(datos[f"Laws {mascara}"], 99)
        for datos in muestras
        for mascara in mascaras_laws
    )


    #------------------ Organizacion de las comparaciones --------------

    familias = {

        "Textura GLCM": [
            ("Grises", "gris")
        ] + [
            (f"GLCM {angulo}", "glcm")
            for angulo in angulos
        ],

        "Textura LBP": [
            ("Grises", "gris"),
            ("LBP", "lbp"),
            ("Histograma LBP", "hist")
        ],

        "Textura Gabor": [
            ("Grises", "gris")
        ] + [
            (f"Gabor {angulo}", "gabor")
            for angulo in angulos
        ],

        "Textura Laws": [
            ("Grises", "gris")
        ] + [
            (f"Laws {mascara}", "laws")
            for mascara in mascaras_laws
        ]
    }


    #------------------ Visualizacion de las cuatro tecnicas -----------

    for nombre_familia, columnas in familias.items():

        for inicio in range(0, 8, 4):

            fig, ax = plt.subplots(
                4,
                len(columnas),
                figsize=(3.2 * len(columnas), 12),
                squeeze=False,
                layout="constrained"
            )

            fig.suptitle(
                nombre_familia,
                fontsize=16
            )

            for fila, datos in enumerate(muestras[inicio:inicio + 4]):

                for columna, (nombre, tipo) in enumerate(columnas):

                    eje = ax[fila, columna]
                    imagen = datos[nombre]

                    if tipo == "gris":

                        eje.imshow(
                            imagen,
                            cmap="gray",
                            vmin=0,
                            vmax=255
                        )
                        eje.axis("off")

                    elif tipo == "glcm":

                        eje.imshow(
                            imagen,
                            cmap="magma",
                            vmin=0,
                            vmax=vmax_glcm,
                            interpolation="nearest"
                        )
                        eje.set_xticks([0, 5, 10, 15])
                        eje.set_yticks([0, 5, 10, 15])

                    elif tipo == "lbp":

                        eje.imshow(
                            imagen,
                            cmap="tab10",
                            vmin=0,
                            vmax=9,
                            interpolation="nearest"
                        )
                        eje.axis("off")

                    elif tipo == "hist":

                        eje.bar(
                            np.arange(10),
                            imagen,
                            color="gray"
                        )
                        eje.set_xticks([0, 3, 6, 9])
                        eje.set_ylim(0, 1)

                    elif tipo == "gabor":

                        eje.imshow(
                            imagen,
                            cmap="gray",
                            vmin=0,
                            vmax=vmax_gabor
                        )
                        eje.axis("off")

                    elif tipo == "laws":

                        eje.imshow(
                            imagen,
                            cmap="gray",
                            vmin=0,
                            vmax=vmax_laws
                        )
                        eje.axis("off")

                    eje.set_title(
                        f"{datos['clase']} - {nombre}",
                        fontsize=9
                    )

            plt.show()


#------------------ Verificacion con varias ROI por tela --------------

verificar_varias_rois = False

if mostrar_texturas and verificar_varias_rois:

    muestras_verificacion = []

    for n in range(1, 9):

        clase = f"Tela {n}"
        archivos = sorted((ruta_rois / clase).glob("*.png"))

        if not archivos:
            raise FileNotFoundError(f"No hay ROI en {clase}")

        # Agrupamos las cuatro ROI segun su fotografia de origen.
        por_fotografia = {}

        for archivo in archivos:
            origen = archivo.stem.rsplit("_ROI_", 1)[0]
            por_fotografia.setdefault(origen, []).append(archivo)

        origenes = sorted(por_fotografia)

        # Seleccionamos hasta cuatro fotografias distintas por clase.
        indices = np.linspace(
            0, len(origenes) - 1,
            min(4, len(origenes)),
            dtype=int
        )

        ejemplos = []

        for indice in indices:

            candidatos = sorted(por_fotografia[origenes[indice]])

            # Utilizamos la ROI 1 de cada fotografia, si existe.
            archivo = next(
                (p for p in candidatos if p.stem.endswith("_ROI_1")),
                candidatos[0]
            )

            imagen = cv.imdecode(
                np.fromfile(archivo, dtype=np.uint8),
                cv.IMREAD_GRAYSCALE
            )

            if imagen is None:
                raise ValueError(f"No se pudo leer {archivo}")

            glcm = aplicar_glcm(imagen)["GLCM 0"]
            _, hist_lbp = aplicar_lbp(imagen)
            gabor = aplicar_gabor(imagen)["Gabor 0"]
            laws = aplicar_laws(imagen)["Laws L5E5"]

            ejemplos.append({
                "archivo": archivo,
                "Grises": imagen,
                "GLCM": glcm,
                "LBP": hist_lbp,
                "Gabor": gabor,
                "Laws": laws
            })

        muestras_verificacion.append((clase, ejemplos))

        print(
            f"{clase}: {len(ejemplos)} fotografias distintas verificadas"
        )

    # Escalas compartidas para comparar las ocho clases.
    todos = [
        e
        for _, ejemplos in muestras_verificacion
        for e in ejemplos
    ]

    vmax_glcm = max(np.max(e["GLCM"]) for e in todos)
    vmax_gabor = max(np.percentile(e["Gabor"], 99) for e in todos)
    vmax_laws = max(np.percentile(e["Laws"], 99) for e in todos)

    for clase, ejemplos in muestras_verificacion:

        fig, ax = plt.subplots(
            len(ejemplos), 5,
            figsize=(17, 3 * len(ejemplos)),
            squeeze=False,
            layout="constrained"
        )

        fig.suptitle(
            f"{clase}: comparacion entre fotografias distintas",
            fontsize=15
        )

        titulos = [
            "Grises",
            "GLCM 0 grados",
            "Histograma LBP",
            "Gabor 0 grados",
            "Laws L5E5"
        ]

        for columna, titulo in enumerate(titulos):
            ax[0, columna].set_title(titulo, fontsize=11)

        for fila, e in enumerate(ejemplos):

            ax[fila, 0].imshow(
                e["Grises"], cmap="gray", vmin=0, vmax=255
            )

            ax[fila, 1].imshow(
                e["GLCM"], cmap="magma",
                vmin=0, vmax=vmax_glcm
            )

            ax[fila, 2].bar(
                np.arange(10), e["LBP"], color="gray"
            )
            ax[fila, 2].set_ylim(0, 1)
            ax[fila, 2].set_xticks([0, 3, 6, 9])

            ax[fila, 3].imshow(
                e["Gabor"], cmap="gray",
                vmin=0, vmax=vmax_gabor
            )

            ax[fila, 4].imshow(
                e["Laws"], cmap="gray",
                vmin=0, vmax=vmax_laws
            )

            ax[fila, 0].set_ylabel(f"Captura {fila + 1}")

            for columna in (0, 1, 3, 4):
                ax[fila, columna].set_xticks([])
                ax[fila, columna].set_yticks([])

        plt.show()



#-----------------------------------------------------------------------
#---------- 1.2.3 Construccion de descriptores numericos ---------------
#-----------------------------------------------------------------------

mostrar_descriptores = True

if mostrar_descriptores:

    if not mostrar_texturas:
        raise RuntimeError(
            "Activa mostrar_texturas para calcular las caracteristicas."
        )

    descriptores = []

    for datos in muestras:

        clase = datos["clase"]


        #------------------ Descriptores GLCM --------------------------

        # Reconstruimos las cuatro matrices en el formato requerido
        # por graycoprops: niveles x niveles x distancias x angulos.

        matrices_glcm = np.stack(
            [
                datos[f"GLCM {angulo}"]
                for angulo in angulos
            ],
            axis=-1
        )[:, :, np.newaxis, :]

        propiedades_glcm = {}

        for propiedad in [
            "contrast",
            "homogeneity",
            "energy",
            "correlation"
        ]:

            propiedades_glcm[propiedad] = graycoprops(
                matrices_glcm,
                propiedad
            )[0]


        #------------------ Descriptor LBP ----------------------------

        # Histograma normalizado de los diez codigos uniformes.
        histograma_lbp = datos["Histograma LBP"].copy()


        #------------------ Descriptores Gabor ------------------------

        estadisticas_gabor = {}

        for angulo in angulos:

            mapa = datos[f"Gabor {angulo}"]

            estadisticas_gabor[angulo] = {
                "media": float(np.mean(mapa)),
                "desviacion": float(np.std(mapa))
            }


        #------------------ Descriptores Laws -------------------------

        estadisticas_laws = {}

        for mascara in mascaras_laws:

            mapa = datos[f"Laws {mascara}"]

            estadisticas_laws[mascara] = {
                "media": float(np.mean(mapa)),
                "desviacion": float(np.std(mapa))
            }


        #------------------ Almacenamiento de resultados --------------

        resultado = {
            "clase": clase,
            "GLCM": propiedades_glcm,
            "LBP": histograma_lbp,
            "Gabor": estadisticas_gabor,
            "Laws": estadisticas_laws
        }

        descriptores.append(resultado)


    #------------------ Comparacion numerica entre telas --------------

    print("\nCOMPARACION DE DESCRIPTORES NUMERICOS")
    print("-" * 84)

    print(
        f"{'Clase':<10}"
        f"{'Contraste':>12}"
        f"{'Homogeneidad':>15}"
        f"{'LBP codigo 9':>15}"
        f"{'Gabor 0 media':>16}"
        f"{'Laws L5E5':>16}"
    )

    print("-" * 84)

    for resultado in descriptores:

        contraste = np.mean(
            resultado["GLCM"]["contrast"]
        )

        homogeneidad = np.mean(
            resultado["GLCM"]["homogeneity"]
        )

        lbp_9 = resultado["LBP"][9]

        gabor_0 = resultado["Gabor"][0]["media"]

        laws_l5e5 = resultado["Laws"]["L5E5"]["media"]

        print(
            f"{resultado['clase']:<10}"
            f"{contraste:>12.3f}"
            f"{homogeneidad:>15.3f}"
            f"{lbp_9:>15.3f}"
            f"{gabor_0:>16.3f}"
            f"{laws_l5e5:>16.3f}"
        )


    #------------------ Ejemplo completo de una ROI -------------------

    ejemplo = descriptores[0]

    print(f"\nEJEMPLO: {ejemplo['clase']}")
    print("-" * 50)

    print("\nGLCM:")

    for propiedad, valores in ejemplo["GLCM"].items():
        print(
            f"{propiedad}: "
            f"{np.round(valores, 4)}"
        )

    print("\nLBP:")
    print(np.round(ejemplo["LBP"], 4))

    print("\nGabor:")

    for angulo, valores in ejemplo["Gabor"].items():
        print(
            f"{angulo} grados: "
            f"media={valores['media']:.4f}, "
            f"desviacion={valores['desviacion']:.4f}"
        )

    print("\nLaws:")

    for mascara, valores in ejemplo["Laws"].items():
        print(
            f"{mascara}: "
            f"media={valores['media']:.4f}, "
            f"desviacion={valores['desviacion']:.4f}"
        )




#-----------------------------------------------------------------------
#---------- 1.2.4 Representacion mediante vectores ---------------------
#------------------ de caracteristicas --------------------------------
#-----------------------------------------------------------------------

mostrar_vectores = True

if mostrar_vectores:

    if not mostrar_descriptores:
        raise RuntimeError(
            "Activa mostrar_descriptores para calcular los vectores."
        )

    vectores = []
    etiquetas = []

    for resultado in descriptores:

        #------------------ Vector GLCM -------------------------------

        vector_glcm = []

        for propiedad in [
            "contrast",
            "homogeneity",
            "energy",
            "correlation"
        ]:
            vector_glcm.extend(
                resultado["GLCM"][propiedad]
            )


        #------------------ Vector LBP --------------------------------

        vector_lbp = resultado["LBP"]


        #------------------ Vector Gabor ------------------------------

        vector_gabor = []

        for angulo in angulos:

            vector_gabor.append(
                resultado["Gabor"][angulo]["media"]
            )

            vector_gabor.append(
                resultado["Gabor"][angulo]["desviacion"]
            )


        #------------------ Vector Laws -------------------------------

        vector_laws = []

        for mascara in mascaras_laws:

            vector_laws.append(
                resultado["Laws"][mascara]["media"]
            )

            vector_laws.append(
                resultado["Laws"][mascara]["desviacion"]
            )


        #------------------ Concatenacion -----------------------------

        vector = np.concatenate([
            np.asarray(vector_glcm, dtype=np.float64),
            np.asarray(vector_lbp, dtype=np.float64),
            np.asarray(vector_gabor, dtype=np.float64),
            np.asarray(vector_laws, dtype=np.float64)
        ])

        # Verificacion de dimension y valores validos
        if len(vector) != 46:
            raise ValueError(
                f"Dimension incorrecta en {resultado['clase']}"
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError(
                f"Valores no validos en {resultado['clase']}"
            )

        vectores.append(vector)
        etiquetas.append(resultado["clase"])


    #------------------ Matriz de caracteristicas ---------------------

    matriz_caracteristicas = np.vstack(vectores)

    print("\nREPRESENTACION MEDIANTE VECTORES")
    print("-" * 55)

    print(f"Cantidad de ROI: {matriz_caracteristicas.shape[0]}")
    print(f"Dimension por ROI: {matriz_caracteristicas.shape[1]}")

    print("\nDimension por tecnica:")
    print("GLCM: 16")
    print("LBP:   10")
    print("Gabor:  8")
    print("Laws:  12")
    print("Total: 46")


    #------------------ Ejemplo de vector ----------------------------

    print(f"\nEJEMPLO: {etiquetas[0]}")
    print("-" * 55)

    print(np.round(matriz_caracteristicas[0], 4))

    print(
        f"\nLongitud del vector: "
        f"{len(matriz_caracteristicas[0])}"
    )


    #------------------ Verificacion de todas las telas ---------------

    print("\nDIMENSION DE LOS VECTORES POR TELA")
    print("-" * 55)

    for clase, vector in zip(etiquetas, vectores):
        print(f"{clase}: {len(vector)} caracteristicas")

#---------------------- Problema 1.3 -------------------------------------
#---------------------- Problema 1.3 -------------------------------------

def estrategia_A(X_train, X_test, y_train, y_test):
    # Instanciar el clasificador común (Random Forest)
    clasificador = RandomForestClassifier(random_state=42)
    
    # Entrenar con los datos aplanados
    clasificador.fit(X_train, y_train)
    
    # Predecir y evaluar
    yp = clasificador.predict(X_test)
    acc = accuracy_score(y_test, yp)
    print('Estrategia A (Flatten):', acc)
    return clasificador, acc

def estrategia_B(X_train, X_test, y_train, y_test):
    # Aplicar PCA para reducir dimensionalidad (reteniendo el 95% de la varianza)
    # Nota: se usa n_components (con 'n_') y random_state no es un parámetro directo de PCA en scikit-learn, 
    # pero podemos fijar svd_solver o simplemente usar svd_solver='full' si es necesario.
    pca = PCA(n_components=0.95, svd_solver='full')
    
    # Ajustar solo con train para evitar data leakage y transformar ambos conjuntos
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"Estrategia B - Componentes seleccionadas por PCA: {pca.n_components_}")
    
    # Entrenar EL MISMO clasificador con los datos reducidos
    clasificador = RandomForestClassifier(random_state=42)
    clasificador.fit(X_train_pca, y_train)
    
    yp = clasificador.predict(X_test_pca)
    acc = accuracy_score(y_test, yp)
    print('Estrategia B (PCA):', acc)
    return clasificador, acc

def estrategia_C(X_descriptors_train, X_descriptors_test, y_train, y_test):
    # Entrenar EL MISMO clasificador utilizando las características diseñadas del Ítem 1.2
    clasificador = RandomForestClassifier(random_state=42)
    clasificador.fit(X_descriptors_train, y_train)
    
    yp = clasificador.predict(X_descriptors_test)
    acc = accuracy_score(y_test, yp)
    print('Estrategia C (Características diseñadas):', acc)
    return clasificador, acc

def start(muestras, matriz_caracteristicas):
    # 1. Preparar los datos para las Estrategias A y B (Imagen original aplanada)
    X_imagenes = []
    y = []
    for i in muestras:
        # Usamos la imagen en grises o redimensionada que tengas guardada en tus diccionarios
        m = i['Grises'] 
        X_imagenes.append(m.ravel())
        y.append(i['clase'])

    X_imagenes = np.array(X_imagenes, dtype=np.float32)

    # Codificar las etiquetas de texto a números enteros
    clases = sorted(set(y))
    mapeo = {c: i for i, c in enumerate(clases)}
    yn = np.array([mapeo[c] for c in y], dtype=np.int32)

    # 2. Realizar la partición UNA SOLA VEZ para las Estrategias A y B 
    # (Garantiza exactamente las mismas muestras de entrenamiento y prueba)
    xt_A, xtest_A, yt_A, ytest_A = train_test_split(
        X_imagenes, yn, test_size=0.2, random_state=42
    )

    print("\n--- EJECUTANDO ESTRATEGIAS ---")
    estrategia_A(xt_A, xtest_A, yt_A, ytest_A)
    estrategia_B(xt_A, xtest_A, yt_A, ytest_A)

    # 3. Para la Estrategia C, utilizamos la matriz de características del Ítem 1.2 (`matriz_caracteristicas`)
    # Usamos exactamente el mismo random_state=42 para mantener la misma partición lógica
    xt_C, xtest_C, yt_C, ytest_C = train_test_split(
        matriz_caracteristicas, yn, test_size=0.2, random_state=42
    )
    
    estrategia_C(xt_C, xtest_C, yt_C, ytest_C)
    
    return None

# Ejecución principal llamando a start (asegúrate de tener calculada `matriz_caracteristicas` del Ítem 1.2)
if mostrar_vectores:
    start(muestras, matriz_caracteristicas)