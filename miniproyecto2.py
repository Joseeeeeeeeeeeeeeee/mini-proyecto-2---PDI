import csv
from collections import Counter
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support,)
from sklearn.model_selection import train_test_split

RUTA_BASE = Path(__file__).resolve().parent
DATASET = RUTA_BASE / "Dataset"
RUTA_ROIS = RUTA_BASE / "ROIs"
SALIDA = RUTA_BASE / "Resultados_1_4_ordenados"

CLASES = [f"Tela {n}" for n in range(1, 9)]
TAMANO_ROI = (180, 180)  # ancho, alto
SEMILLA = 42
PROPORCION_PRUEBA = 0.20

# Las pruebas visuales del 1.2 son opcionales; no cambian el entrenamiento.
REGENERAR_ROIS = False
MOSTRAR_ROIS_1_2 = False
MOSTRAR_CANALES_1_2 = False
MOSTRAR_REALCES_1_2 = False
MOSTRAR_TEXTURAS_1_2 = False
MOSTRAR_VARIAS_ROIS_1_2 = False

# Los graficos del 1.4 siempre se guardan; esta variable controla ventanas.
MOSTRAR_GRAFICOS_1_4 = True

ANGULOS = (0, 45, 90, 135)
PROPIEDADES_GLCM = ("contrast", "homogeneity", "energy", "correlation")
MASCARAS_LAWS = ("L5E5", "E5L5", "L5S5", "S5S5", "W5W5", "R5R5")

VECTORES_LAWS = {
    "L5": np.array([1, 4, 6, 4, 1], dtype=np.float32),
    "E5": np.array([-1, -2, 0, 2, 1], dtype=np.float32),
    "S5": np.array([-1, 0, 2, 0, -1], dtype=np.float32),
    "W5": np.array([-1, 2, 0, -2, 1], dtype=np.float32),
    "R5": np.array([1, -4, 6, -4, 1], dtype=np.float32),
}


def encabezado(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


def leer_imagen(ruta, modo=cv.IMREAD_COLOR):
    imagen = cv.imdecode(np.fromfile(str(ruta), dtype=np.uint8), modo)
    if imagen is None:
        raise ValueError(f"No se pudo leer la imagen: {ruta}")
    return imagen


#=======================================================================
#       1.1 - ADQUISICION Y CONSTRUCCION DEL DATASET                     #
#=======================================================================


def resumen_imagenes_originales():
    if not DATASET.is_dir():
        print("Aviso: falta Dataset/. Se utilizaran las ROI existentes.")
        return

    encabezado("1.1 - FOTOGRAFIAS ORIGINALES")
    total = 0

    for clase in CLASES:
        carpeta = DATASET / clase
        archivos = sorted(
            ruta for ruta in carpeta.iterdir()
            if ruta.is_file() and ruta.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ) if carpeta.is_dir() else []

        print(f"{clase}: {len(archivos)} fotografias")
        total += len(archivos)

    print(f"Total de fotografias: {total}")


#=======================================================================
#       1.2.1 - PREPARACION DE IMAGENES Y ROI                            #
#=======================================================================


def extraer_rois(ruta):
    original_bgr = leer_imagen(ruta)
    original = cv.cvtColor(original_bgr, cv.COLOR_BGR2RGB)

    alto, ancho = original.shape[:2]
    lado = min(alto, ancho) // 4

    centros = [
        (ancho // 4, alto // 4),
        (3 * ancho // 4, alto // 4),
        (ancho // 4, 3 * alto // 4),
        (3 * ancho // 4, 3 * alto // 4),
    ]

    rois = []
    cuadros = []

    for cx, cy in centros:
        x0, y0 = cx - lado // 2, cy - lado // 2
        rois.append(original[y0:y0 + lado, x0:x0 + lado].copy())
        cuadros.append((x0, y0, lado))

    return original, rois, cuadros


def guardar_rois():
    if not DATASET.is_dir():
        raise FileNotFoundError(f"No existe {DATASET}")

    encabezado("1.2.1 - EXTRACCION Y GUARDADO DE ROI")

    for clase in CLASES:
        entrada = DATASET / clase
        destino = RUTA_ROIS / clase
        destino.mkdir(parents=True, exist_ok=True)

        archivos = sorted(
            p for p in entrada.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        for archivo in archivos:
            _, rois, _ = extraer_rois(archivo)

            for num, roi_rgb in enumerate(rois, start=1):
                roi_bgr = cv.cvtColor(roi_rgb, cv.COLOR_RGB2BGR)
                salida = destino / f"{archivo.stem}_ROI_{num}.png"

                correcto, datos = cv.imencode(".png", roi_bgr)

                if not correcto:
                    raise RuntimeError(f"No se pudo guardar {salida}")

                datos.tofile(str(salida))

        print(f"{clase}: {len(archivos) * 4} ROI guardadas")


def mostrar_recorte_y_canales():
    for clase in CLASES:
        originales = sorted((DATASET / clase).glob("*"))

        if MOSTRAR_ROIS_1_2 and originales:
            originales = [
                p for p in originales
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]

            if originales:
                original, rois, cuadros = extraer_rois(originales[0])
                marcada = original.copy()

                for x, y, lado in cuadros:
                    cv.rectangle(
                        marcada,
                        (x, y),
                        (x + lado, y + lado),
                        (255, 0, 0),
                        5
                    )

                fig, ax = plt.subplots(
                    1, 5,
                    figsize=(16, 3.5),
                    num=f"1.2.1 - Recortes {clase}"
                )

                for eje, imagen, titulo in zip(
                    ax,
                    [marcada] + rois,
                    ["Captura + ROI"] + [f"ROI {i}" for i in range(1, 5)]
                ):
                    eje.imshow(imagen)
                    eje.set_title(titulo)
                    eje.axis("off")

                fig.tight_layout()

        if MOSTRAR_CANALES_1_2:
            archivos = sorted((RUTA_ROIS / clase).glob("*.png"))

            if not archivos:
                raise FileNotFoundError(f"Faltan ROI en {clase}")

            imagen_bgr = leer_imagen(archivos[0])
            rgb = cv.cvtColor(imagen_bgr, cv.COLOR_BGR2RGB)
            gris = cv.cvtColor(imagen_bgr, cv.COLOR_BGR2GRAY)
            r, g, b = cv.split(rgb)

            fig, ax = plt.subplots(
                1, 5,
                figsize=(15, 3.5),
                num=f"1.2.1 - Canales {clase}"
            )

            for eje, imagen, titulo in zip(
                ax,
                [rgb, r, g, b, gris],
                ["RGB", "R", "G", "B", "Grises"]
            ):
                eje.imshow(
                    imagen,
                    cmap=None if titulo == "RGB" else "gray"
                )
                eje.set_title(titulo)
                eje.axis("off")

            fig.tight_layout()


def comparar_realces():
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

        # Transformaciones puntuales
        brillo = np.clip(gris + 30, 0, 255).astype(np.uint8)
        contraste = np.clip(
            1.35 * (gris - 128) + 128, 0, 255
        ).astype(np.uint8)
        negativo = cv.bitwise_not(imagen_gris)

        logaritmica = np.clip(
            255 * np.log1p(gris) / np.log(256), 0, 255
        ).astype(np.uint8)

        gamma_06 = np.clip(
            255 * (gris / 255) ** 0.6, 0, 255
        ).astype(np.uint8)

        gamma_16 = np.clip(
            255 * (gris / 255) ** 1.6, 0, 255
        ).astype(np.uint8)

        tramos = np.interp(
            imagen_gris,
            [0, 60, 170, 255],
            [0, 40, 220, 255]
        ).astype(np.uint8)

        # Histograma
        estiramiento = cv.normalize(
            imagen_gris, None, 0, 255, cv.NORM_MINMAX
        )
        ecualizacion = cv.equalizeHist(imagen_gris)

        ahe = cv.createCLAHE(
            clipLimit=0.0,
            tileGridSize=(8, 8)
        ).apply(imagen_gris)

        clahe = cv.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        ).apply(imagen_gris)

        # Filtros espaciales
        promedio = cv.blur(imagen_gris, (3, 3))
        gaussiano = cv.GaussianBlur(imagen_gris, (3, 3), 0)
        mediana = cv.medianBlur(imagen_gris, 3)

        kernel_3 = np.ones((3, 3), dtype=np.uint8)
        minimo = cv.erode(imagen_gris, kernel_3)
        maximo = cv.dilate(imagen_gris, kernel_3)

        # Operadores de gradiente
        roberts_x = np.array(
            [[1, 0], [0, -1]], dtype=np.float32
        )
        roberts_y = np.array(
            [[0, 1], [-1, 0]], dtype=np.float32
        )

        prewitt_x = np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=np.float32)

        prewitt_y = np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ], dtype=np.float32)

        roberts = magnitud_gradiente(
            imagen_gris, roberts_x, roberts_y
        )
        prewitt = magnitud_gradiente(
            imagen_gris, prewitt_x, prewitt_y
        )

        sobel = normalizar_respuesta(cv.magnitude(
            cv.Sobel(imagen_gris, cv.CV_32F, 1, 0, ksize=3),
            cv.Sobel(imagen_gris, cv.CV_32F, 0, 1, ksize=3)
        ))

        scharr = normalizar_respuesta(cv.magnitude(
            cv.Scharr(imagen_gris, cv.CV_32F, 1, 0),
            cv.Scharr(imagen_gris, cv.CV_32F, 0, 1)
        ))

        laplaciano = cv.Laplacian(
            imagen_gris, cv.CV_32F, ksize=1
        )

        mapa_laplaciano = normalizar_respuesta(laplaciano)

        realce_laplaciano = np.clip(
            gris - laplaciano, 0, 255
        ).astype(np.uint8)

        suavizada = cv.GaussianBlur(
            imagen_gris, (3, 3), 0
        )

        respuesta_log = cv.Laplacian(
            suavizada, cv.CV_32F, ksize=1
        )

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

    grupos_tecnicas = {
        'Transformaciones de intensidad I': [
            'Grises',
            'Brillo +30',
            'Contraste 1.35',
            'Negativo',
            'Logaritmica'
        ],
        'Transformaciones de intensidad II': [
            'Grises',
            'Gamma 0.6',
            'Gamma 1.6',
            'Por tramos'
        ],
        'Procesamiento de histogramas': [
            'Grises',
            'Estiramiento',
            'Ecualizacion',
            'AHE',
            'CLAHE'
        ],
        'Filtros espaciales de suavizado': [
            'Grises',
            'Promedio 3x3',
            'Gaussiano 3x3',
            'Mediana 3x3',
            'Minimo 3x3',
            'Maximo 3x3'
        ],
        'Operadores de gradiente': [
            'Grises',
            'Roberts',
            'Prewitt',
            'Sobel',
            'Scharr'
        ],
        'Laplaciano y detectores de bordes': [
            'Grises',
            'Laplaciano',
            'Realce Laplaciano',
            'LoG (magnitud)',
            'Canny'
        ]
    }

    muestras = []

    for n in range(1, 9):
        clase = f'Tela {n}'
        archivos = sorted(
            (RUTA_ROIS / clase).glob('*.png')
        )

        if not archivos:
            raise FileNotFoundError(f'No hay ROI en {clase}')

        imagen = cv.imdecode(
            np.fromfile(archivos[0], dtype=np.uint8),
            cv.IMREAD_COLOR
        )

        if imagen is None:
            raise ValueError(
                f'No se pudo leer {archivos[0]}'
            )

        imagen_gris = cv.cvtColor(
            imagen, cv.COLOR_BGR2GRAY
        )

        muestras.append((
            clase,
            aplicar_tecnicas(imagen_gris)
        ))

    for nombre_grupo, columnas in grupos_tecnicas.items():
        for inicio in range(0, len(muestras), 4):
            grupo = muestras[inicio:inicio + 4]

            fig, ax = plt.subplots(
                len(grupo),
                len(columnas),
                figsize=(3.3 * len(columnas), 3 * len(grupo)),
                squeeze=False,
                layout='constrained'
            )

            fig.suptitle(nombre_grupo, fontsize=15)

            for fila, (clase, resultados) in enumerate(grupo):
                for columna, tecnica in enumerate(columnas):
                    ax[fila, columna].imshow(
                        resultados[tecnica],
                        cmap='gray',
                        vmin=0,
                        vmax=255
                    )
                    ax[fila, columna].set_title(
                        f'{clase} - {tecnica}',
                        fontsize=9
                    )
                    ax[fila, columna].axis('off')

            plt.show()


#=======================================================================
#       1.2.2 - EXTRACCION DE CARACTERISTICAS DE TEXTURA                 #
#=======================================================================


def aplicar_glcm(imagen_gris):
    matriz = graycomatrix(
        imagen_gris // 16,
        distances=[1],
        angles=np.deg2rad(ANGULOS),
        levels=16,
        symmetric=True,
        normed=True,
    )

    return matriz


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
    )[:10]

    histograma = histograma.astype(np.float64)
    histograma /= histograma.sum()

    return lbp, histograma


def aplicar_gabor(imagen_gris):
    mapas = {}

    for angulo in ANGULOS:
        kernel = cv.getGaborKernel(
            (21, 21),
            sigma=5,
            theta=np.deg2rad(angulo),
            lambd=8,
            gamma=0.5,
            psi=0,
            ktype=cv.CV_32F
        )

        kernel = kernel - kernel.mean()
        kernel = kernel / (
            np.sum(np.abs(kernel)) + 1e-8
        )

        respuesta = cv.filter2D(
            imagen_gris.astype(np.float32),
            cv.CV_32F,
            kernel,
            borderType=cv.BORDER_REFLECT
        )

        mapas[angulo] = cv.GaussianBlur(
            np.abs(respuesta),
            (7, 7),
            0
        )

    return mapas


def aplicar_laws(imagen_gris):
    fondo = cv.GaussianBlur(
        imagen_gris.astype(np.float32),
        (15, 15),
        0
    )

    centrada = imagen_gris.astype(np.float32) - fondo
    mapas = {}

    for nombre in MASCARAS_LAWS:
        kernel = np.outer(
            VECTORES_LAWS[nombre[:2]],
            VECTORES_LAWS[nombre[2:]]
        ).astype(np.float32)

        kernel = kernel / (
            np.sum(np.abs(kernel)) + 1e-8
        )

        respuesta = cv.filter2D(
            centrada,
            cv.CV_32F,
            kernel,
            borderType=cv.BORDER_REFLECT
        )

        mapas[nombre] = cv.GaussianBlur(
            np.abs(respuesta),
            (9, 9),
            0
        )

    return mapas


#=======================================================================
#       1.2.3 - DESCRIPTORES NUMERICOS                                  #
#=======================================================================


def calcular_descriptores(imagen_gris, devolver_mapas=False):
    glcm = aplicar_glcm(imagen_gris)
    lbp, histograma_lbp = aplicar_lbp(imagen_gris)
    gabor = aplicar_gabor(imagen_gris)
    laws = aplicar_laws(imagen_gris)

    # GLCM: 16 valores
    vector_glcm = np.concatenate([
        graycoprops(glcm, propiedad)[0]
        for propiedad in PROPIEDADES_GLCM
    ])

    # LBP: 10 valores
    vector_lbp = histograma_lbp

    # Gabor: 8 valores
    vector_gabor = np.array([
        valor
        for angulo in ANGULOS
        for valor in (
            np.mean(gabor[angulo]),
            np.std(gabor[angulo])
        )
    ])

    # Laws: 12 valores
    vector_laws = np.array([
        valor
        for nombre in MASCARAS_LAWS
        for valor in (
            np.mean(laws[nombre]),
            np.std(laws[nombre])
        )
    ])

    bloques = (
        vector_glcm,
        vector_lbp,
        vector_gabor,
        vector_laws
    )

    vector = np.concatenate(
        bloques
    ).astype(np.float64)

    if vector.shape != (46,) or not np.all(np.isfinite(vector)):
        raise ValueError(
            "El vector de textura no tiene 46 valores numericos validos"
        )

    if devolver_mapas:
        return vector, {
            "glcm": glcm,
            "lbp": lbp,
            "lbp_hist": histograma_lbp,
            "gabor": gabor,
            "laws": laws
        }

    return vector


def mostrar_texturas_1_2(imagen_gris, clase):
    _, mapas = calcular_descriptores(
        imagen_gris,
        devolver_mapas=True
    )

    familias = [
        (
            "GLCM",
            [mapas["glcm"][:, :, 0, i] for i in range(4)],
            [f"{a} grados" for a in ANGULOS],
            "magma"
        ),
        (
            "LBP",
            [mapas["lbp"]],
            ["Patrones locales"],
            "tab10"
        ),
        (
            "Gabor",
            [mapas["gabor"][a] for a in ANGULOS],
            [f"{a} grados" for a in ANGULOS],
            "gray"
        ),
        (
            "Laws",
            [mapas["laws"][m] for m in MASCARAS_LAWS],
            list(MASCARAS_LAWS),
            "gray"
        ),
    ]

    for familia, imagenes, titulos, cmap in familias:
        fig, ax = plt.subplots(
            1,
            len(imagenes) + 1,
            figsize=(3 * (len(imagenes) + 1), 3),
            num=f"1.2.2 {clase}: {familia}"
        )

        ax[0].imshow(
            imagen_gris,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        ax[0].set_title("Grises")
        ax[0].axis("off")

        escala = max(
            np.percentile(mapa, 99)
            for mapa in imagenes
        )

        for eje, imagen, titulo in zip(
            ax[1:],
            imagenes,
            titulos
        ):
            eje.imshow(
                imagen,
                cmap=cmap,
                vmin=0,
                vmax=escala
            )
            eje.set_title(titulo)
            eje.axis("off")

        fig.tight_layout()


def mostrar_varias_fotografias_textura():
    for clase in CLASES:
        archivos = sorted(
            (RUTA_ROIS / clase).glob("*_ROI_1.png")
        )

        indices = np.linspace(
            0,
            len(archivos) - 1,
            min(4, len(archivos)),
            dtype=int
        )

        seleccion = [archivos[i] for i in indices]

        fig, axes = plt.subplots(
            len(seleccion),
            5,
            figsize=(14, 2.8 * len(seleccion)),
            squeeze=False,
            num=f"1.2.2 - Varias fotografias {clase}"
        )

        for fila, ruta in enumerate(seleccion):
            gris = leer_imagen(
                ruta,
                cv.IMREAD_GRAYSCALE
            )

            _, mapas = calcular_descriptores(
                gris,
                devolver_mapas=True
            )

            elementos = [
                (gris, "Grises", "gray"),
                (
                    mapas["glcm"][:, :, 0, 0],
                    "GLCM 0°",
                    "magma"
                ),
                (mapas["lbp"], "LBP", "tab10"),
                (
                    mapas["gabor"][0],
                    "Gabor 0°",
                    "gray"
                ),
                (
                    mapas["laws"]["L5E5"],
                    "Laws L5E5",
                    "gray"
                ),
            ]

            for col, (imagen, titulo, cmap) in enumerate(elementos):
                axes[fila, col].imshow(
                    imagen,
                    cmap=cmap
                )

                axes[fila, col].set_title(
                    f"{titulo} - {ruta.stem}",
                    fontsize=8
                )

                axes[fila, col].axis("off")

        fig.tight_layout()


#=======================================================================
#       1.2.4 - VECTORES DE CARACTERISTICAS / DATASET COMPLETO          #
#=======================================================================


def imprimir_tabla_descriptores(X_textura, y):

    encabezado("1.2.3 - COMPARACION NUMERICA DE TEXTURAS")

    print(
        f"{'Clase':<10}"
        f"{'Contraste':>12}"
        f"{'Homogeneidad':>15}"
        f"{'LBP uniformes':>16}"
        f"{'LBP no uniformes':>18}"
        f"{'Gabor 0':>13}"
        f"{'Laws L5E5':>14}"
    )

    for clase in CLASES:

        i = int(np.flatnonzero(y == clase)[0])
        v = X_textura[i]

        contraste = np.mean(v[0:4])
        homogeneidad = np.mean(v[4:8])

        # Patrones locales mediante LBP
        patrones_uniformes = np.sum(v[16:25])
        patrones_no_uniformes = v[25]

        gabor_0 = v[26]
        laws_l5e5 = v[34]

        print(
            f"{clase:<10}"
            f"{contraste:>12.3f}"
            f"{homogeneidad:>15.3f}"
            f"{patrones_uniformes:>16.3f}"
            f"{patrones_no_uniformes:>18.3f}"
            f"{gabor_0:>13.3f}"
            f"{laws_l5e5:>14.3f}"
        )

def nombres_descriptores():
    nombres = [
        f"GLCM {propiedad} {angulo}°"
        for propiedad in PROPIEDADES_GLCM
        for angulo in ANGULOS
    ]

    nombres += [
        f"LBP codigo {i}"
        for i in range(10)
    ]

    nombres += [
        f"Gabor {angulo}° {est}"
        for angulo in ANGULOS
        for est in ("media", "desv.")
    ]

    nombres += [
        f"Laws {nombre} {est}"
        for nombre in MASCARAS_LAWS
        for est in ("media", "desv.")
    ]

    if len(nombres) != 46:
        raise RuntimeError(
            "Cantidad incorrecta de nombres de descriptores"
        )

    return nombres


def generar_tabla_descriptores(X_textura, nombres):

    encabezado("1.2.4 - TABLA DE DESCRIPTORES Y VECTOR DE EJEMPLO")

    SALIDA.mkdir(parents=True, exist_ok=True)

    # Cantidades obtenidas de la configuracion real del codigo
    n_glcm = len(PROPIEDADES_GLCM) * len(ANGULOS)
    n_gabor = len(ANGULOS) * 2
    n_laws = len(MASCARAS_LAWS) * 2
    n_total = X_textura.shape[1]
    n_lbp = n_total - n_glcm - n_gabor - n_laws

    nombres_vector = nombres_descriptores()

    if n_total != len(nombres_vector):
        raise ValueError(
            "La cantidad de nombres no coincide con el vector."
        )

    angulos = ", ".join(f"{a}°" for a in ANGULOS)
    mascaras = ", ".join(MASCARAS_LAWS)

    traducciones = {
        "contrast": "Contraste",
        "homogeneity": "Homogeneidad",
        "energy": "Energia",
        "correlation": "Correlacion"
    }

    tabla = []

    # GLCM: una fila por cada propiedad
    for propiedad in PROPIEDADES_GLCM:

        tabla.append([
            "GLCM",
            traducciones.get(propiedad, propiedad),
            angulos,
            len(ANGULOS)
        ])

    # LBP: histograma de patrones locales
    tabla.append([
        "LBP",
        "Histograma de patrones locales",
        "P=8, R=1, uniforme",
        n_lbp
    ])

    # Gabor
    tabla.append([
        "Gabor",
        "Media y desviacion estandar",
        angulos,
        n_gabor
    ])

    # Laws
    tabla.append([
        "Laws",
        "Media y desviacion estandar",
        mascaras,
        n_laws
    ])

    # Total
    tabla.append([
        "TOTAL",
        "Vector de caracteristicas",
        "Estrategia C",
        n_total
    ])

    #---------------------------------------------------------------
    # TABLA RESUMEN EN TERMINAL
    #---------------------------------------------------------------

    print(
        f"\n{'Tecnica':<12}"
        f"{'Descriptor':<36}"
        f"{'Configuracion':<47}"
        f"{'Valores':>8}"
    )

    print("-" * 103)

    for tecnica, descriptor, configuracion, cantidad in tabla:

        print(
            f"{tecnica:<12}"
            f"{descriptor:<36}"
            f"{configuracion:<47}"
            f"{cantidad:>8}"
        )

    #---------------------------------------------------------------
    # GUARDAR TABLA RESUMEN
    #---------------------------------------------------------------

    ruta_tabla = SALIDA / "tabla_descriptores_1_2_4.csv"

    with ruta_tabla.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as archivo:

        escritor = csv.writer(archivo, delimiter=";")

        escritor.writerow([
            "Tecnica",
            "Descriptor",
            "Configuracion",
            "Cantidad de valores"
        ])

        escritor.writerows(tabla)

    #---------------------------------------------------------------
    # EJEMPLO REAL: PRIMERA ROI DEL DATASET
    #---------------------------------------------------------------

    indice = 0

    nombre_roi = nombres[indice]
    vector = X_textura[indice]

    print("\n" + "=" * 70)
    print("EJEMPLO REAL DEL VECTOR DE CARACTERISTICAS")
    print("=" * 70)

    print(f"ROI seleccionada: {nombre_roi}")
    print(f"Dimension del vector: {len(vector)}")

    print(
        f"\n{'Posicion':<12}"
        f"{'Descriptor':<38}"
        f"{'Valor':>15}"
    )

    print("-" * 65)

    filas_ejemplo = []

    for posicion, (descriptor, valor) in enumerate(
        zip(nombres_vector, vector),
        start=1
    ):

        print(
            f"{posicion:<12}"
            f"{descriptor:<38}"
            f"{valor:>15.6f}"
        )

        filas_ejemplo.append([
            nombre_roi,
            posicion,
            descriptor,
            float(valor)
        ])

    #---------------------------------------------------------------
    # GUARDAR LOS 46 VALORES CALCULADOS
    #---------------------------------------------------------------

    ruta_ejemplo = SALIDA / "ejemplo_vector_1_2_4.csv"

    with ruta_ejemplo.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as archivo:

        escritor = csv.writer(archivo, delimiter=";")

        escritor.writerow([
            "ROI",
            "Posicion",
            "Descriptor",
            "Valor"
        ])

        escritor.writerows(filas_ejemplo)

    print("\nArchivos generados:")
    print(ruta_tabla)
    print(ruta_ejemplo)


def construir_dataset():
    encabezado("1.2.4 - CONSTRUCCION DEL DATASET COMPLETO")

    X_pixeles = []
    X_textura = []
    y = []
    grupos = []
    nombres = []

    for clase in CLASES:
        carpeta = RUTA_ROIS / clase

        if not carpeta.is_dir():
            raise FileNotFoundError(
                f"No existe la carpeta: {carpeta}"
            )

        archivos = sorted(
            p for p in carpeta.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        if not archivos:
            raise FileNotFoundError(
                f"No hay imagenes ROI en: {carpeta}"
            )

        for ruta in archivos:
            partes = ruta.stem.rsplit("_ROI_", 1)

            if (
                len(partes) != 2
                or partes[1] not in {"1", "2", "3", "4"}
            ):
                raise ValueError(
                    f"Nombre de ROI inesperado: {ruta.name}"
                )

            origen = f"{clase}/{partes[0]}"

            gris = leer_imagen(
                ruta,
                cv.IMREAD_GRAYSCALE
            )

            if gris.shape != (
                TAMANO_ROI[1],
                TAMANO_ROI[0]
            ):
                gris = cv.resize(
                    gris,
                    TAMANO_ROI,
                    interpolation=cv.INTER_AREA
                )

            X_pixeles.append(
                gris.ravel().astype(np.float32)
            )

            X_textura.append(
                calcular_descriptores(gris)
            )

            y.append(clase)
            grupos.append(origen)
            nombres.append(str(ruta.relative_to(RUTA_ROIS)))

        print(
            f"{clase}: {len(archivos)} ROI procesadas",
            flush=True
        )

    X_pixeles = np.asarray(
        X_pixeles,
        dtype=np.float32
    )

    X_textura = np.asarray(
        X_textura,
        dtype=np.float64
    )

    y, grupos, nombres = map(
        np.asarray,
        (y, grupos, nombres)
    )

    fotos, conteos = np.unique(
        grupos,
        return_counts=True
    )

    if not np.all(conteos == 4):
        raise ValueError(
            "Cada fotografia de origen debe aportar cuatro ROI"
        )

    if (
        X_pixeles.shape != (len(y), 32400)
        or X_textura.shape != (len(y), 46)
    ):
        raise ValueError(
            "Dimensiones incorrectas de las representaciones"
        )

    if not np.all(np.isfinite(X_textura)):
        raise ValueError(
            "Los vectores de textura contienen valores invalidos"
        )

    print(f"Fotografias originales: {len(fotos)}")
    print(
        f"ROI: {len(y)} | "
        f"Flatten: {X_pixeles.shape} | "
        f"Textura: {X_textura.shape}"
    )

    print(
        "Descriptores: GLCM=16, LBP=10, Gabor=8, Laws=12. Total=46"
    )

    print(
        f"Ejemplo {nombres[0]}: "
        f"{np.round(X_textura[0], 4)}"
    )

    return X_pixeles, X_textura, y, grupos, nombres


#=======================================================================
#       1.3 - CONSTRUCCION Y ENTRENAMIENTO DE 3 ESTRATEGIAS             #
#=======================================================================


def dividir_por_fotografia(y, grupos):
    fotos = np.unique(grupos)
    clases_fotos = []

    for foto in fotos:
        clases = np.unique(y[grupos == foto])

        if len(clases) != 1:
            raise ValueError(
                f"La fotografia {foto} corresponde a varias clases"
            )

        clases_fotos.append(clases[0])

    fotos_train, fotos_test = train_test_split(
        fotos,
        test_size=PROPORCION_PRUEBA,
        random_state=SEMILLA,
        stratify=clases_fotos
    )

    idx_train = np.flatnonzero(
        np.isin(grupos, fotos_train)
    )

    idx_test = np.flatnonzero(
        np.isin(grupos, fotos_test)
    )

    if not set(fotos_train).isdisjoint(fotos_test):
        raise RuntimeError(
            "Filtracion de fotografias entre train y test"
        )

    if len(idx_train) + len(idx_test) != len(y):
        raise RuntimeError(
            "Se perdieron ROI en la particion"
        )

    if (
        set(y[idx_train]) != set(CLASES)
        or set(y[idx_test]) != set(CLASES)
    ):
        raise RuntimeError(
            "Faltan clases en train o test"
        )

    encabezado("1.3 - DIVISION POR FOTOGRAFIA (80/20)")

    print(
        f"Fotos: train={len(fotos_train)}, "
        f"test={len(fotos_test)}"
    )

    print(
        f"ROI: train={len(idx_train)}, "
        f"test={len(idx_test)}"
    )

    print("Fotografias compartidas: 0")

    print(
        f"{'Clase':<10}"
        f"{'Fotos train':>14}"
        f"{'Fotos test':>13}"
        f"{'ROI train':>12}"
        f"{'ROI test':>11}"
    )

    for clase in CLASES:
        n_train_fotos = sum(
            f.startswith(clase + "/")
            for f in fotos_train
        )

        n_test_fotos = sum(
            f.startswith(clase + "/")
            for f in fotos_test
        )

        n_train_roi = np.sum(
            y[idx_train] == clase
        )

        n_test_roi = np.sum(
            y[idx_test] == clase
        )

        print(
            f"{clase:<10}"
            f"{n_train_fotos:>14}"
            f"{n_test_fotos:>13}"
            f"{n_train_roi:>12}"
            f"{n_test_roi:>11}"
        )

    return idx_train, idx_test


def clasificador_comun():
    return RandomForestClassifier(
        n_estimators=100,
        random_state=SEMILLA,
        n_jobs=-1
    )


class PCAEnEspacioDeMuestras:
    def __init__(self, varianza_objetivo=0.95):
        self.varianza_objetivo = varianza_objetivo

    def fit_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = len(X)

        self.mean_ = X.mean(axis=0)
        centrado = X - self.mean_

        gram = (
            centrado @ centrado.T
        ) / (n - 1)

        autovalores, autovectores = np.linalg.eigh(gram)

        orden = np.argsort(
            autovalores
        )[::-1]

        autovalores = np.maximum(
            autovalores[orden], 0
        )

        autovectores = autovectores[:, orden]

        if autovalores.sum() <= 0:
            raise ValueError(
                "Imagenes sin variabilidad para PCA"
            )

        proporciones = autovalores / autovalores.sum()

        k = int(
            np.searchsorted(
                np.cumsum(proporciones),
                self.varianza_objetivo
            ) + 1
        )

        tolerancia = (
            max(autovalores[0], 1.0) * 1e-12
        )

        if k > np.count_nonzero(
            autovalores > tolerancia
        ):
            raise ValueError(
                "Componentes PCA numericamente nulas"
            )

        self.n_components_ = k
        self.explained_variance_ratio_ = proporciones[:k]

        singular = np.sqrt(
            autovalores[:k] * (n - 1)
        )

        self.components_ = (
            (
                centrado.T
                @ autovectores[:, :k]
            ) / singular
        ).T

        return autovectores[:, :k] * singular

    def transform(self, X):
        return (
            np.asarray(X, dtype=np.float64) - self.mean_
        ) @ self.components_.T


def entrenar_estrategias(
    X_pixeles,
    X_textura,
    y,
    idx_train,
    idx_test
):
    """A=Flatten; B=Flatten+PCA; C=Textura."""

    encabezado("1.3 - ENTRENAMIENTO DE LAS TRES ESTRATEGIAS")

    y_train = y[idx_train]
    y_test = y[idx_test]

    X_train = X_pixeles[idx_train]
    X_test = X_pixeles[idx_test]

    # Estrategia A - Flatten
    modelo_A = clasificador_comun()
    modelo_A.fit(X_train, y_train)
    pred_A = modelo_A.predict(X_test)

    # Estrategia B - PCA
    pca = PCAEnEspacioDeMuestras(
        varianza_objetivo=0.95
    )

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    modelo_B = clasificador_comun()
    modelo_B.fit(X_train_pca, y_train)
    pred_B = modelo_B.predict(X_test_pca)

    # Estrategia C - Textura
    modelo_C = clasificador_comun()

    modelo_C.fit(
        X_textura[idx_train],
        y_train
    )

    pred_C = modelo_C.predict(
        X_textura[idx_test]
    )

    print(
        f"A - Flatten (32400): "
        f"{accuracy_score(y_test, pred_A):.2%}"
    )

    print(
        f"B - PCA ({pca.n_components_}): "
        f"{accuracy_score(y_test, pred_B):.2%}"
    )

    print(
        f"Varianza retenida en entrenamiento: "
        f"{pca.explained_variance_ratio_.sum():.2%}"
    )

    print(
        f"C - Textura (46): "
        f"{accuracy_score(y_test, pred_C):.2%}"
    )

    return {
        "modelos": {
            "A": modelo_A,
            "B": modelo_B,
            "C": modelo_C
        },
        "pca": pca,
        "y_test": y_test,
        "predicciones": {
            "A": pred_A,
            "B": pred_B,
            "C": pred_C
        },
        "idx_test": idx_test
    }


#=======================================================================
#       1.4 - EVALUACION Y COMPARACION DE RESULTADOS                    #
#=======================================================================


def guardar_figura(figura, nombre):
    SALIDA.mkdir(parents=True, exist_ok=True)

    figura.savefig(
        SALIDA / nombre,
        dpi=170,
        bbox_inches="tight"
    )

    if not MOSTRAR_GRAFICOS_1_4:
        plt.close(figura)


def figuras_comparacion(resumen):
    fig, ax = plt.subplots(
        figsize=(8.5, 4.5),
        num="1.4 - Comparacion general"
    )

    x = np.arange(3)

    barras_acc = ax.bar(
        x - 0.19,
        [r["accuracy"] * 100 for r in resumen],
        0.38,
        label="Accuracy"
    )

    barras_f1 = ax.bar(
        x + 0.19,
        [r["f1_macro"] * 100 for r in resumen],
        0.38,
        label="F1 macro"
    )

    ax.set_xticks(
        x,
        ["A - Flatten", "B - PCA", "C - Textura"]
    )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Porcentaje (%)")
    ax.set_title("Comparacion de desempeno")

    ax.bar_label(
        barras_acc,
        fmt="%.1f",
        padding=2
    )

    ax.bar_label(
        barras_f1,
        fmt="%.1f",
        padding=2
    )

    ax.legend()
    fig.tight_layout()

    guardar_figura(
        fig,
        "01_comparacion_metricas.png"
    )


def figuras_confusion(matrices):
    fig, axes = plt.subplots(
        1, 3,
        figsize=(20, 6),
        num="1.4 - Matrices de confusion A, B y C"
    )

    for eje, clave in zip(axes, "ABC"):
        ConfusionMatrixDisplay(
            matrices[clave],
            display_labels=CLASES
        ).plot(
            ax=eje,
            values_format="d",
            colorbar=False,
            xticks_rotation=50
        )

        eje.set_title(f"Estrategia {clave}")
        eje.tick_params(
            axis="both",
            labelsize=8
        )

    fig.tight_layout()

    guardar_figura(
        fig,
        "02_matrices_confusion.png"
    )


def elegir_ejemplos(
    y_real,
    y_pred,
    nombres,
    correctos,
    maximo=4
):
    indices = np.flatnonzero(
        (y_real == y_pred) == correctos
    )

    conteos = Counter(
        y_real[i] for i in indices
    )

    clases_ordenadas = sorted(
        conteos,
        key=lambda c: (-conteos[c], c)
    )

    seleccion = []
    fotos_usadas = set()

    for clase in clases_ordenadas:
        for i in indices:
            if y_real[i] != clase:
                continue

            ruta = Path(str(nombres[i]))
            foto = (
                ruta.parent.name,
                ruta.stem.rsplit("_ROI_", 1)[0]
            )

            if foto not in fotos_usadas:
                seleccion.append(i)
                fotos_usadas.add(foto)
                break

        if len(seleccion) == maximo:
            break

    for i in indices:
        if len(seleccion) == maximo:
            break

        ruta = Path(str(nombres[i]))
        foto = (
            ruta.parent.name,
            ruta.stem.rsplit("_ROI_", 1)[0]
        )

        if foto not in fotos_usadas:
            seleccion.append(i)
            fotos_usadas.add(foto)

    return seleccion


def figura_ejemplos(
    y_real,
    y_pred,
    nombres,
    estrategia
):
    aciertos = elegir_ejemplos(
        y_real,
        y_pred,
        nombres,
        True
    )

    errores = elegir_ejemplos(
        y_real,
        y_pred,
        nombres,
        False
    )

    fig, axes = plt.subplots(
        2, 4,
        figsize=(13, 6.6),
        num=f"1.4 - Ejemplos estrategia {estrategia}"
    )

    for fila, (tipo, ejemplos) in enumerate((
        ("ACIERTO", aciertos),
        ("ERROR", errores)
    )):
        for col, ax in enumerate(axes[fila]):
            ax.axis("off")

            if col >= len(ejemplos):
                continue

            i = ejemplos[col]
            archivo = RUTA_ROIS / str(nombres[i])

            imagen = leer_imagen(
                archivo,
                cv.IMREAD_GRAYSCALE
            )

            ax.imshow(
                imagen,
                cmap="gray",
                vmin=0,
                vmax=255
            )

            ax.set_title(
                f"{tipo} | Real: {y_real[i]}\n"
                f"Predicha: {y_pred[i]}",
                fontsize=9
            )

            ax.set_xlabel(
                archivo.name,
                fontsize=8
            )

            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()

    guardar_figura(
        fig,
        f"03_ejemplos_{estrategia}.png"
    )


def figura_importancias(importancias):
    nombres = nombres_descriptores()

    orden = np.argsort(
        importancias
    )[::-1]

    top = orden[:12][::-1]

    fig, ax = plt.subplots(
        figsize=(9.5, 6),
        num="1.4 - Importancia de los descriptores C"
    )

    ax.barh(
        np.arange(len(top)),
        importancias[top]
    )

    ax.set_yticks(
        np.arange(len(top)),
        [nombres[i] for i in top]
    )

    ax.set_xlabel(
        "Importancia relativa (disminucion de impureza)"
    )

    ax.set_title(
        "Estrategia C - 12 descriptores mas importantes"
    )

    fig.tight_layout()

    guardar_figura(
        fig,
        "04_importancia_descriptores_C.png"
    )


def guardar_resultados_csv(
    resumen,
    y_real,
    predicciones,
    nombres
):
    with (
        SALIDA / "resumen_metricas.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=list(resumen[0].keys())
        )

        escritor.writeheader()
        escritor.writerows(resumen)

    with (
        SALIDA / "predicciones_ROI_prueba.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as archivo:

        escritor = csv.writer(archivo)

        escritor.writerow([
            "ROI",
            "Clase_real",
            "Pred_A",
            "Pred_B",
            "Pred_C",
            "A_correcta",
            "B_correcta",
            "C_correcta"
        ])

        for i, nombre in enumerate(nombres):
            escritor.writerow([
                nombre,
                y_real[i],
                *(predicciones[c][i] for c in "ABC"),
                *(
                    int(y_real[i] == predicciones[c][i])
                    for c in "ABC"
                )
            ])


def evaluar_estrategias(
    resultados,
    nombres_test
):
    encabezado("1.4 - EVALUACION Y COMPARACION")

    SALIDA.mkdir(exist_ok=True)

    y_real = resultados["y_test"]
    predicciones = resultados["predicciones"]

    resumen = []
    matrices = {}

    #---------------------------------------------------------------
    # 1.4.1 - Metricas y confusion por clase
    #---------------------------------------------------------------

    for clave in "ABC":
        y_pred = predicciones[clave]

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_real,
                y_pred,
                labels=CLASES,
                average="macro",
                zero_division=0
            )
        )

        p_w, r_w, f_w, _ = (
            precision_recall_fscore_support(
                y_real,
                y_pred,
                labels=CLASES,
                average="weighted",
                zero_division=0
            )
        )

        acc = accuracy_score(
            y_real,
            y_pred
        )

        resumen.append({
            "estrategia": clave,
            "accuracy": acc,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "precision_weighted": p_w,
            "recall_weighted": r_w,
            "f1_weighted": f_w
        })

        matrices[clave] = confusion_matrix(
            y_real,
            y_pred,
            labels=CLASES
        )

        print(f"\nESTRATEGIA {clave}")

        print(
            classification_report(
                y_real,
                y_pred,
                labels=CLASES,
                digits=3,
                zero_division=0
            )
        )

        pares = Counter(
            (real, pred)
            for real, pred in zip(y_real, y_pred)
            if real != pred
        )

        print("Confusiones mas frecuentes:")

        for (real, pred), n in pares.most_common(5):
            print(
                f"  {real} -> {pred}: {n} ROI"
            )

    #---------------------------------------------------------------
    # 1.4.2 - Tabla resumida de metricas y dimensiones
    #---------------------------------------------------------------

    print("\nCOMPARACION GLOBAL")

    print(
        f"{'Estrategia':<12}"
        f"{'Accuracy':>12}"
        f"{'Precision':>13}"
        f"{'Recall':>12}"
        f"{'F1 macro':>12}"
    )

    for r in resumen:
        print(
            f"{r['estrategia']:<12}"
            f"{r['accuracy']:>12.2%}"
            f"{r['precision_macro']:>13.2%}"
            f"{r['recall_macro']:>12.2%}"
            f"{r['f1_macro']:>12.2%}"
        )

    dimensiones = {
        "A": 32400,
        "B": resultados["pca"].n_components_,
        "C": 46
    }

    print("\nDIMENSIONES Y REDUCCION RESPECTO A FLATTEN")

    print(
        f"{'Estrategia':<12}"
        f"{'Dimensiones':>14}"
        f"{'Reduccion':>14}"
    )

    for clave, cantidad in dimensiones.items():
        reduccion = 100 * (
            1 - cantidad / dimensiones["A"]
        )

        print(
            f"{clave:<12}"
            f"{cantidad:>14}"
            f"{reduccion:>13.2f}%"
        )

    print(
        "A: 180 x 180 x 1 = 32400 valores; "
        f"B retiene {resultados['pca'].explained_variance_ratio_.sum():.2%} "
        "de varianza; C: 16+10+8+12 = 46"
    )

    #---------------------------------------------------------------
    # 1.4.3 - Relevancia de descriptores de textura
    #---------------------------------------------------------------

    importancias = resultados[
        "modelos"
    ]["C"].feature_importances_

    if importancias.shape != (46,):
        raise RuntimeError(
            "Se esperaban 46 importancias de textura"
        )

    print("\nIMPORTANCIA DE CARACTERISTICAS DE TEXTURA")

    for familia, a, b in (
        ("GLCM", 0, 16),
        ("LBP", 16, 26),
        ("Gabor", 26, 34),
        ("Laws", 34, 46)
    ):
        print(
            f"{familia:<7} "
            f"Suma={sum(importancias[a:b]):.4f} "
            f"Promedio={np.mean(importancias[a:b]):.4f}"
        )

    nombres = nombres_descriptores()

    for indice in np.argsort(
        importancias
    )[::-1][:10]:
        print(
            f"  {nombres[indice]}: "
            f"{importancias[indice]:.4f}"
        )

    #---------------------------------------------------------------
    # 1.4.4 - Graficos y archivos
    #---------------------------------------------------------------

    figuras_comparacion(resumen)
    figuras_confusion(matrices)

    for clave in "ABC":
        figura_ejemplos(
            y_real,
            predicciones[clave],
            nombres_test,
            clave
        )

    figura_importancias(importancias)

    guardar_resultados_csv(
        resumen,
        y_real,
        predicciones,
        nombres_test
    )

    print(
        f"\nFiguras y CSV guardados en: {SALIDA}"
    )

    print(
        "NOTA: las cuatro ROI de una misma foto "
        "no son observaciones independientes."
    )

    print(
        "Analizar el bajo soporte de Tela 6 y Tela 8."
    )

    print(
        "La importancia de Random Forest es por impureza, "
        "no causalidad."
    )

    if MOSTRAR_GRAFICOS_1_4:
        plt.show()

    return resumen, matrices, importancias


#=======================================================================
#       EJECUCION PRINCIPAL - UNA SOLA VEZ                              #
#=======================================================================


def main():
    # 1.1 - Lectura y organizacion del dataset
    resumen_imagenes_originales()

    # 1.2.1 - Preparacion de imagenes y ROI
    if REGENERAR_ROIS:
        guardar_rois()

    if MOSTRAR_ROIS_1_2 or MOSTRAR_CANALES_1_2:
        mostrar_recorte_y_canales()

    if MOSTRAR_REALCES_1_2:
        comparar_realces()

    # 1.2.2, 1.2.3 y 1.2.4 - Caracteristicas y vectores
    X_pixeles, X_textura, y, grupos, nombres = construir_dataset()

    imprimir_tabla_descriptores(
        X_textura,
        y
    )

    generar_tabla_descriptores(
    X_textura,
    nombres
    )

    if MOSTRAR_TEXTURAS_1_2:
        for clase in CLASES:
            indice = int(
                np.flatnonzero(y == clase)[0]
            )

            gris = X_pixeles[indice].reshape(
                TAMANO_ROI[1],
                TAMANO_ROI[0]
            ).astype(np.uint8)

            mostrar_texturas_1_2(
                gris,
                clase
            )

    if MOSTRAR_VARIAS_ROIS_1_2:
        mostrar_varias_fotografias_textura()

    # 1.3 - Particion y entrenamiento
    idx_train, idx_test = dividir_por_fotografia(
        y,
        grupos
    )

    resultados = entrenar_estrategias(
        X_pixeles,
        X_textura,
        y,
        idx_train,
        idx_test
    )

    # 1.4 - Evaluacion y visualizacion de resultados
    evaluar_estrategias(
        resultados,
        nombres[idx_test]
    )


if __name__ == "__main__":
    main()
