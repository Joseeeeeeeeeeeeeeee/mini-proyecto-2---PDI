
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

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


#------------------ Realce de contraste CLAHE -------------------------

def aplicar_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):

    clahe = cv.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(img)


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





#------------------ Representacion de color ----------------------------

mostrar_colores = False

if mostrar_colores:

    fig = plt.figure(figsize=(13, 14))

    fig.suptitle(
        "Comparacion RGB y escala de grises",
        fontsize=18,
        y=0.97
    )

    # 4 filas y 2 grupos de telas por fila
    rejilla = fig.add_gridspec(
        4, 2,
        left=0.04,
        right=0.96,
        bottom=0.04,
        top=0.92,
        wspace=0.25,
        hspace=0.45
    )

    for n in range(8):

        clase = f"Tela {n+1}"

        archivos = sorted((ruta_rois / clase).glob("*.png"))

        if not archivos:
            raise FileNotFoundError(f"No hay ROI en {clase}")

        imagen = cv.imdecode(
            np.fromfile(archivos[0], dtype=np.uint8),
            cv.IMREAD_COLOR
        )

        imagen_RGB = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)
        imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)

        fila = n // 2
        columna = n % 2

        # Dos imagenes juntas dentro de cada grupo
        par = rejilla[fila, columna].subgridspec(
            1, 2,
            wspace=0.03
        )

        ax_RGB = fig.add_subplot(par[0, 0])
        ax_gris = fig.add_subplot(par[0, 1])

        # Imagen RGB
        ax_RGB.imshow(imagen_RGB)
        ax_RGB.set_title(
            f"{clase} - RGB",
            fontsize=11,
            pad=3
        )
        ax_RGB.axis("off")

        # Escala de grises
        ax_gris.imshow(
            imagen_gris,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        ax_gris.set_title(
            "Escala de grises",
            fontsize=11,
            pad=3
        )
        ax_gris.axis("off")

    plt.show()


#------------------ Comparacion de realce CLAHE ------------------------

mostrar_clahe = True

if mostrar_clahe:

    clase = "Tela 1"

    archivos = sorted((ruta_rois / clase).glob("*.png"))

    archivo = archivos[0]

    imagen = cv.imdecode(
        np.fromfile(archivo, dtype=np.uint8),
        cv.IMREAD_COLOR
    )

    imagen_RGB = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)
    imagen_gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)

    imagen_clahe = aplicar_clahe(imagen_gris)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(imagen_RGB)
    ax[0].set_title("Original RGB")
    ax[0].axis("off")

    ax[1].imshow(imagen_gris, cmap="gray", vmin=0, vmax=255)
    ax[1].set_title("Escala de grises")
    ax[1].axis("off")

    ax[2].imshow(imagen_clahe, cmap="gray", vmin=0, vmax=255)
    ax[2].set_title("CLAHE")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()



#-----------------------------------------------------------------------
#---------- 1.2.2 Identificacion y extraccion de -----------------------
#------------------ caracteristicas visuales ---------------------------
#-----------------------------------------------------------------------




#-----------------------------------------------------------------------
#---------- 1.2.3 Construccion de descriptores numericos ---------------
#-----------------------------------------------------------------------




#-----------------------------------------------------------------------
#---------- 1.2.4 Representacion mediante vectores ---------------------
#------------------ de caracteristicas --------------------------------
#-----------------------------------------------------------------------

