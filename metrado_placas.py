import os
import io
import argparse
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image


def dividir_en_cuadros(imagen, filas=3, columnas=3):
    ancho, alto = imagen.size
    ancho_corte = ancho // columnas
    alto_corte = alto // filas
    subimagenes = []
    for i in range(filas):
        for j in range(columnas):
            caja = (j * ancho_corte, i * alto_corte, (j + 1) * ancho_corte, (i + 1) * alto_corte)
            subimagenes.append(imagen.crop(caja))
    return subimagenes

def cargar_api_key():
    env_path = os.path.join(os.path.dirname(__file__), "GOOGLE_API_KEY.env")
    load_dotenv(env_path)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("API key not found in GOOGLE_API_KEY.env")
    return api_key

def guardar_salida(texto, ruta):
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"✅ Saved: {ruta}")

def leer_archivo(ruta):
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def convertir_a_partes(imagenes_pil):
    partes = []
    for img in imagenes_pil:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        partes.append({"inline_data": {"mime_type": "image/png", "data": buffer.getvalue()}})
    return partes

class AgenteGeminiMultimodal:
    def __init__(self, api_key, modelo="gemini-2.5-pro-preview-06-05", temperatura=0.3):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(modelo)
        self.historial = []
        self.temperatura = temperatura

    def enviar(self, prompt_texto, lista_imagenes):
        entrada = {"role": "user", "parts": [prompt_texto] + lista_imagenes}
        self.historial.append(entrada)
        respuesta = self.model.generate_content(self.historial)
        salida = respuesta.text.strip()
        self.historial.append({"role": "model", "parts": [salida]})
        return salida


def main():
    parser = argparse.ArgumentParser(description="Gemini Quantity Takeoff Pipeline")
    parser.add_argument("--planos_dir", required=True, help="Root directory for construction drawings.")
    parser.add_argument("--output_dir", default="results", help="Directory to save outputs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    api_key = cargar_api_key()
    agente = AgenteGeminiMultimodal(api_key)

    def cargar_imagen(nombre_archivo):
        ruta = os.path.join(args.planos_dir, nombre_archivo)
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"Image not found: {ruta}")
        return Image.open(ruta)

    def cargar_imagenes_en_carpetas(directorio_base, carpetas_objetivo, extensiones_permitidas=None):
        if extensiones_permitidas is None:
            extensiones_permitidas = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

        imagenes = []
        for carpeta in carpetas_objetivo:
            ruta_carpeta = os.path.join(directorio_base, carpeta)
            if not os.path.isdir(ruta_carpeta):
                print(f"Carpeta no encontrada: {ruta_carpeta}")
                continue
            for file in os.listdir(ruta_carpeta):
                if any(file.lower().endswith(ext) for ext in extensiones_permitidas):
                    ruta_img = os.path.join(ruta_carpeta, file)
                    try:
                        img = Image.open(ruta_img)
                        imagenes.append((ruta_img, img))
                    except Exception as e:
                        print(f"Error cargando {ruta_img}: {e}")
        return imagenes

    respuestas = []
    historial_prompt = []
    #################################################################################################
    # PASO 0 - Especificaciones para placas.

    carpetas_objetivo_0 = [
        "E-01_Edificio Multifamiliar _Especificaciones Técnicas",
    ]
    imagenes_info = cargar_imagenes_en_carpetas(args.planos_dir, carpetas_objetivo_0)
    imagenes = [img for ruta, img in imagenes_info]

    # Divide en crops y agrupa
    imagenes_con_crops = []
    for img in imagenes:
        crops = dividir_en_cuadros(img)
        imagenes_con_crops.extend([img] + crops)

    # Convierte a formato final para el modelo
    prompt_0 = (
        "You are a civil engineer specialized in quantity takeoff and construction measurement (metrado).\n"
        "Focus exclusively on extracting all technical specifications, codes, and construction notes related to slabs (placas) from the provided drawing.\n\n"
        "Specifically, identify and organize:\n"
        "- Cuadro de solapado (splicing/overlapping schedule) for slabs: include any requirements on minimum overlaps, bar diameters, types, and notes about splicing locations.\n"
        "- All details and specifications for estribos (stirrups): spacing, diameter, placement, anchorage requirements, and any related notes.\n"
        "- Technical notes, construction instructions, and any special considerations affecting slab construction or measurement (e.g., placement conditions, sequencing, materials to be used, joint requirements, construction tolerances, etc.).\n"
        "- Any codes, references, or standards cited in the drawing for slab execution.\n\n"
        "Present the information in a **structured and clear technical summary**, using bullet points or a table if necessary. DO NOT include unrelated data or general geometry (such as length or width), focus only on technical specifications and detailed construction notes for slabs."
    )
    resultado_0 = agente.enviar(prompt_0, imagenes_con_crops)
    respuestas.append(resultado_0)
    historial_prompt.append(prompt_0)
    guardar_salida(prompt_0, os.path.join(args.output_dir, "step_0_prompt.txt"))
    guardar_salida(resultado_0, os.path.join(args.output_dir, "step_0_response.txt"))
    ###########################################################################################
    #################################################################################################
    # PASO 1 - Cement type per floor from X and arrow annotations
    carpetas_objetivo_1 = [
        "E-06_Edificio Multifamiliar_Placa-I",
        "E-07_Edificio Multifamiliar_Placa-II",
        "E-08_Edificio Multifamiliar_Placa-III"
    ]
    imagenes_info = cargar_imagenes_en_carpetas(args.planos_dir, carpetas_objetivo_1)
    imagenes = [img for ruta, img in imagenes_info]

    # Divide en crops y agrupa
    imagenes_con_crops = []
    for img in imagenes:
        crops = dividir_en_cuadros(img)
        imagenes_con_crops.extend([img] + crops)
    contexto_1 = respuestas[-1]
    # Convierte a formato final para el modelo
    prompt_1 = (
        "You are a civil engineer with expertise in quantity takeoff and construction measurement (metrado).\n"
        "Your task is to analyze the provided structural drawing and extract ALL relevant information needed to accurately quantify and measure slabs (placas), such as P1, P2, etc.\n\n"
        "For each slab identified, extract and organize the following data in a structured table:\n"
        "- Slab code or name (e.g., P1, P2, etc.)\n"
        "- Width (m)\n"
        "- Length (m)\n"
        "- Thickness (m or cm)\n"
        "- Concrete type or class (if shown)\n"
        "- Reinforcement details (number of rods, diameter, spacing, type, if present)\n"
        "- Location/floor (if specified)\n"
        "- Save the perimeter in a column, you will use this later as the supposed WIDTH of the slab formwork"
        "- Any notes, callouts, or references that affect measurement or construction (such as slopes, openings, additional layers, special features)\n\n"
        "Be precise and include ALL relevant details found in the drawing that are needed for a complete slab quantity takeoff.\n"
        "Return ONLY the structured table with the requested data, and make sure no critical information is omitted."
    )
    resultado_1 = agente.enviar(prompt_1 + "\n\n" + contexto_1, imagenes_con_crops)
    respuestas.append(resultado_1)
    historial_prompt.append(prompt_1)
    guardar_salida(prompt_1, os.path.join(args.output_dir, "step_1_prompt.txt"))
    guardar_salida(resultado_1, os.path.join(args.output_dir, "step_1_response.txt"))
    ###########################################################################################
    # PASO 2 - Width and length of slab
    carpetas_objetivo_2 = [
        "E-06_Edificio Multifamiliar_Placa-I",
        "E-07_Edificio Multifamiliar_Placa-II",
        "E-08_Edificio Multifamiliar_Placa-III"
    ]
    imagenes_info = cargar_imagenes_en_carpetas(args.planos_dir, carpetas_objetivo_2)
    imagenes = [img for ruta, img in imagenes_info]
    # Divide en crops y agrupa
    imagenes_con_crops = []
    for img in imagenes:
        crops = dividir_en_cuadros(img)
        imagenes_con_crops.extend([img] + crops)

    imagenes_2 = imagenes_con_crops
    contexto_2 = respuestas[-1]
    prompt_2 = (
        "You are a civil engineer with professional expertise in construction quantity takeoff, structural drawing interpretation, and the preparation of technical reports for building projects.\n"
        "Analyze the provided foundation plan (planta de cimentación) carefully.\n\n"
        "Your main task is to identify and record which slabs (placas) are associated with which sections or cuts (cortes) in the plan. Cross-reference any slab codes (e.g., P1, P2) with their corresponding section/cut designations, as indicated in the drawing.\n\n"
        "Present your findings in a structured table with the following columns:\n"
        "- Slab Code (e.g., P1, P2, etc.)\n"
        "- Associated Section/Cut Code (e.g., Corte-1, Corte-2, etc.)\n"
        "- Any technical notes or references that clarify the association\n\n"
        "Be thorough and precise. Use all available callouts, section markers, legends, or drawing references. Only include associations that are clearly indicated in the plan."
    )

    resultado_2 = agente.enviar(prompt_2 + "\n\n" + contexto_2, imagenes_2)
    respuestas.append(resultado_2)
    historial_prompt.append(prompt_2)
    guardar_salida(prompt_2, os.path.join(args.output_dir, "step_2_prompt.txt"))
    guardar_salida(resultado_2, os.path.join(args.output_dir, "step_2_response.txt"))

    # PASO 3 - Height of slab per floor
    img_3 = cargar_imagen("alturas.png")
    crops_3 = dividir_en_cuadros(img_3)
    imagenes_3 = convertir_a_partes([img_3] + crops_3)
    contexto_3 = "\n\n".join(respuestas)
    prompt_3 = (
        "You are a civil engineer specialized in quantity takeoff and reinforced concrete detailing.\n"
        "Analyze the provided structural image and use any previous data as needed.\n\n"
        "Your main tasks are:\n"
        "1. Determine the exact height of each slab (placa) on every floor, based on building height, floor levels, and all dimensional annotations shown.\n"
        "2. Using these heights and slab geometry, calculate the required length of reinforcing bars (varillas) for each slab. Be sure to consider and specify:\n"
        "   - Any required overlaps (solapes) for splicing\n"
        "   - Deductions or additions due to floor gaps, slab supports, or other construction features\n"
        "   - Precise start and end points of bars\n"
        "3. Calculate the number and total length of stirrups (estribos) needed for each slab, according to the provided technical details, spacing, and slab dimensions. State clearly:\n"
        "   - The spacing and diameter used for each slab\n"
        "   - Any adjustments based on slab geometry, overlaps, or required anchorage\n\n"
        "Return all results in a clear and structured table, showing for each slab:\n"
        "- Slab code/name\n"
        "- Floor/level\n"
        "- Height (m)\n"
        "- Total length of main reinforcement bars (m)\n"
        "- Number of bars (if possible)\n"
        "- Total number and length of stirrups (estribos)\n"
        "- Any relevant calculation notes (e.g., deductions, overlaps, construction allowances)\n\n"
        "Carefully sum or subtract any dimensions as indicated by the drawing or technical notes (e.g., subtracting slab thickness, adding overlap lengths, etc.). If any data is missing or unclear, make a note in the table."
    )

    resultado_3 = agente.enviar(prompt_3 + "\n\n" + contexto_3, imagenes_3)
    respuestas.append(resultado_3)
    historial_prompt.append(prompt_3)
    guardar_salida(prompt_3, os.path.join(args.output_dir, "step_3_prompt.txt"))
    guardar_salida(resultado_3, os.path.join(args.output_dir, "step_3_response.txt"))

    # PASO 4 - Complete Excel table with everything
    img_4 = cargar_imagen("base.png")
    imagenes_4 = convertir_a_partes([img_4])
    contexto_4 = "\n\n".join(respuestas)
    prompt_4 = (
        "You are a civil engineer who specializes in automated quantity take-offs.\n"
        "\n"
        "TASK\n"
        "-----\n"
        "Using the information from the previous table 'Reinforcement Steel Quantity Takeoff for Structural Walls (Placas)', fill the final summary table as follows:\n"
        "\n"
        "- For each element of type 'C' (Concrete or Main Reinforcement), use the LENGTH, WIDTH, HEIGHT, and steel quantities as previously calculated.\n"
        "- For each element of type 'E' (Estribo/Encofrado), set:\n"
        "    - ALTO = Total height of the slab\n"
        "    - ANCHO = 1.00 (always)\n"
        "    - LARGO = Perimeter of the slab (taken from the 'Perimeter' column in previous steps)\n"
        "    - Fierro Cant., Fierro Long, Tipo de Fierro = Values as previously calculated for the stirrups\n"
        "- If a slab or story has multiple estribo types, include a row for each unique combination.\n"
        "\n"
        "Return the plain-text table in the following format (NO explanations, NO markdown):\n"
        "\n"
        "ELEMENTO|TIPO|ALTO|ANCHO|LARGO|Fierro Cant.|Fierro Long|Tipo de Fierro (1/4,3/8,etc)\n"
        "\n"
        "IMPORTANT:\n"
        "- For 'E' rows, be sure that LARGO is the slab perimeter (not the width or thickness) and ANCHO is always 1.00. Do NOT use any other dimension.\n"
        "- Do not invent or omit any data.\n"
    )


    resultado_4 = agente.enviar(prompt_4 + "\n\n" + contexto_4, imagenes_4)
    guardar_salida(prompt_4, os.path.join(args.output_dir, "step_4_prompt.txt"))
    guardar_salida(resultado_4, os.path.join(args.output_dir, "step_4_response.txt"))


if __name__ == "__main__":
    main()
