# Se ejecuta el script interno para generar el archivo descargable .docx para el usuario.
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

doc = docx.Document()

# Configurar márgenes estrechos
sections = doc.sections
for section in sections:
    section.top_margin = Inches(0.5)
    section.bottom_margin = Inches(0.5)
    section.left_margin = Inches(0.5)
    section.right_margin = Inches(0.5)

# Título y encabezado
title_p = doc.add_paragraph()
title_p.paragraph_format.space_after = Pt(12)
run_title = title_p.add_run("PLANTILLA DE REPASO: MÉTODO CORNELL")
run_title.font.name = 'Arial'
run_title.font.size = Pt(18)
run_title.font.bold = True
run_title.font.color.rgb = RGBColor(26, 54, 93)

# Tabla de metadatos (Asignatura, Tema, Fecha)
meta_table = doc.add_table(rows=1, cols=3)
meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
meta_table.autofit = False

meta_data = [
    ("RAMO: Redes de Datos", Inches(3.0)),
    ("TEMA: ", Inches(3.0)),
    ("FECHA: __/__/2026", Inches(1.5))
]

hdr_cells = meta_table.rows[0].cells
for i, (text, width) in enumerate(meta_data):
    hdr_cells[i].text = text
    hdr_cells[i].width = width
    p = hdr_cells[i].paragraphs[0]
    p.runs[0].font.name = 'Arial'
    p.runs[0].font.size = Pt(10)
    p.runs[0].font.bold = True
    # Añadir borde inferior suave a la meta tabla
    tcPr = hdr_cells[i]._tc.get_or_add_tcPr()
    tcBorders = parse_xml(r'<w:tcBorders %s><w:bottom w:val="single" w:sz="6" w:space="0" w:color="A0A0A0"/></w:tcBorders>' % nsdecls('w'))
    tcPr.append(tcBorders)

doc.add_paragraph().paragraph_format.space_after = Pt(6)

# Tabla principal Cornell (2 columnas, 2 filas)
# Fila 1: Notas e ideas clave. Fila 2: Resumen.
main_table = doc.add_table(rows=2, cols=2)
main_table.alignment = WD_TABLE_ALIGNMENT.CENTER
main_table.autofit = False

# Ancho de columnas: Izquierda (Ideas) ~2.25 in, Derecha (Notas) ~5.25 in
col_widths = [Inches(2.25), Inches(5.25)]

# Configurar Fila 1 (Cuerpo)
row1 = main_table.rows[0]
trPr1 = row1._tr.get_or_add_trPr()
trPr1.append(parse_xml(r'<w:trHeight %s w:val="7200" w:hRule="atLeast"/>' % nsdecls('w'))) # Alto mínimo ~12.7 cm

# Celda Izquierda - Preguntas
cell_izq = row1.cells[0]
cell_izq.width = col_widths[0]
p_izq = cell_izq.paragraphs[0]
run_izq_head = p_izq.add_run("IDEAS CLAVE / PREGUNTAS\n")
run_izq_head.font.bold = True
run_izq_head.font.size = Pt(10)
run_izq_head.font.color.rgb = RGBColor(120, 120, 120)
p_izq.add_run("(Escribe aquí tus preguntas de autoexamen basadas en las notas de la derecha)\n\n• ¿...?\n• ¿...?\n• ¿...?")
p_izq.runs[1].font.size = Pt(10)
p_izq.runs[1].font.italic = True
p_izq.runs[1].font.color.rgb = RGBColor(160, 160, 160)

# Celda Derecha - Notas
cell_der = row1.cells[1]
cell_der.width = col_widths[1]
p_der = cell_der.paragraphs[0]
run_der_head = p_der.add_run("NOTAS DE REPASO Y CONCEPTOS\n")
run_der_head.font.bold = True
run_der_head.font.size = Pt(10)
run_der_head.font.color.rgb = RGBColor(120, 120, 120)
p_der.add_run("(Esquemas, definiciones, topologías y comandos clave)\n\n")
p_der.runs[1].font.size = Pt(10)
p_der.runs[1].font.italic = True
p_der.runs[1].font.color.rgb = RGBColor(160, 160, 160)

# Añadir ejemplo de formato de comandos en las notas
p_cmd = cell_der.add_paragraph()
run_cmd_lbl = p_cmd.add_run("Tip para comandos: ")
run_cmd_lbl.font.size = Pt(10)
run_cmd_lbl.font.bold = True
run_cmd_txt = p_cmd.add_run("Router(config)# interface g0/0")
run_cmd_txt.font.name = 'Consolas'
run_cmd_txt.font.size = Pt(10)
run_cmd_txt.font.color.rgb = RGBColor(199, 37, 78)

# Configurar Fila 2 (Resumen - Celdas combinadas)
row2 = main_table.rows[1]
trPr2 = row2._tr.get_or_add_trPr()
trPr2.append(parse_xml(r'<w:trHeight %s w:val="1800" w:hRule="atLeast"/>' % nsdecls('w'))) # Alto mínimo ~3.2 cm

cell_res = row2.cells[0]
cell_res.width = Inches(7.5)
cell_res.merge(row2.cells[1])

p_res = cell_res.paragraphs[0]
run_res_head = p_res.add_run("RESUMEN GENERAL (Sintetiza la hoja en 2 o 3 líneas al terminar de repasar)\n\n")
run_res_head.font.bold = True
run_res_head.font.size = Pt(10)
run_res_head.font.color.rgb = RGBColor(120, 120, 120)

# Aplicar sombreado y bordes limpios estilo técnico a la tabla principal
def style_cell(cell, bg_color=None, top=None, bottom=None, left=None, right=None):
    tcPr = cell._tc.get_or_add_tcPr()
    if bg_color:
        shd = parse_xml(r'<w:shd %s w:fill="%s"/>' % (nsdecls('w'), bg_color))
        tcPr.append(shd)
    
    borders_str = r'<w:tcBorders %s>' % nsdecls('w')
    if top: borders_str += r'<w:top w:val="%s" w:sz="%s" w:space="0" w:color="%s"/>' % top
    else: borders_str += r'<w:top w:val="none"/>'
    if bottom: borders_str += r'<w:bottom w:val="%s" w:sz="%s" w:space="0" w:color="%s"/>' % bottom
    else: borders_str += r'<w:bottom w:val="none"/>'
    if left: borders_str += r'<w:left w:val="%s" w:sz="%s" w:space="0" w:color="%s"/>' % left
    else: borders_str += r'<w:left w:val="none"/>'
    if right: borders_str += r'<w:right w:val="%s" w:sz="%s" w:space="0" w:color="%s"/>' % right
    else: borders_str += r'<w:right w:val="none"/>'
    borders_str += r'</w:tcBorders>'
    tcPr.append(parse_xml(borders_str))

# Aplicar los estilos visuales a cada celda
style_cell(cell_izq, bg_color="F8FAFC", bottom=("single", "12", "CBD5E1"), right=("single", "12", "CBD5E1"))
style_cell(cell_der, bg_color="FFFFFF", bottom=("single", "12", "CBD5E1"))
style_cell(cell_res, bg_color="F1F5F9", top=("single", "12", "CBD5E1"))

# Guardar documento
doc.save("Plantilla_Cornell_Redes.docx")
