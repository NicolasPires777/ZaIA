import tensorflow as tf
import numpy as np
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), 'video-manager'))
from video_manager import extract_frames, extract_frames_with_timestamps

model = tf.keras.models.load_model("models/200epochs.keras")
CLASS_NAMES = ["merchan", "conteudo"]


def classificar_video(video_path, num_frames):
    frames = extract_frames(video_path, num_frames=num_frames)

    frames = np.array(frames, dtype=np.float32) / 255.0

    preds = model.predict(frames)

    resultados = []

    for i, pred in enumerate(preds):
        classe = int(np.argmax(pred))
        confianca = float(np.max(pred))

        nome_classe = CLASS_NAMES[classe]

        resultados.append({
            "frame": i,
            "classe": nome_classe,
            "confianca": confianca
        })

    return resultados

def classificar_frames(frames):
    frames = np.array(frames, dtype=np.float32) / 255.0
    preds = model.predict(frames, verbose=0)

    classes = []
    confiancas = []

    for pred in preds:
        if pred.shape[0] == 1:
            classe = int(pred[0] > 0.5)
            confianca = float(pred[0])
        else:
            classe = int(np.argmax(pred))
            confianca = float(np.max(pred))

        classes.append(classe)
        confiancas.append(confianca)

    return classes, confiancas

def agrupar_segmentos(timestamps, classes):
    segmentos = []

    inicio = timestamps[0]
    classe_atual = classes[0]

    for i in range(1, len(classes)):
        if classes[i] != classe_atual:
            fim = timestamps[i - 1]
            segmentos.append((inicio, fim, classe_atual))

            inicio = timestamps[i]
            classe_atual = classes[i]

    segmentos.append((inicio, timestamps[-1], classe_atual))
    return segmentos

def formatar_tempo(segundos):
    m = int(segundos // 60)
    s = int(segundos % 60)
    return f"{m:02d}:{s:02d}"

def classificar_video_com_trechos(video_path, every_n_seconds=1):
    frames, timestamps = extract_frames_with_timestamps(
        video_path,
        every_n_seconds=every_n_seconds
    )

    classes, _ = classificar_frames(frames)

    classes = estabilizar_classes(classes, min_consecutive=5)

    segmentos = agrupar_segmentos(timestamps, classes)


    resultado_formatado = []
    for inicio, fim, classe in segmentos:
        resultado_formatado.append({
            "inicio": formatar_tempo(inicio),
            "fim": formatar_tempo(fim),
            "classe": classe
        })

    return resultado_formatado

def estabilizar_classes(classes, min_consecutive=3):
    """
    Só troca de classe se a nova classe aparecer
    em min_consecutive frames consecutivos
    """
    if not classes:
        return []

    classe_atual = classes[0]
    classes_estaveis = [classe_atual]

    contagem = 0
    candidata = None

    for c in classes[1:]:
        if c == classe_atual:
            # Continua igual → zera candidato
            contagem = 0
            candidata = None
            classes_estaveis.append(classe_atual)
        else:
            # Possível troca
            if candidata is None or c != candidata:
                candidata = c
                contagem = 1
            else:
                contagem += 1

            if contagem >= min_consecutive:
                # Confirma troca
                classe_atual = candidata
                contagem = 0
                candidata = None

            classes_estaveis.append(classe_atual)

    return classes_estaveis



### Classificar video
# resultados = classificar_video("/home/nicolas/Videos/Treinamento-IA/teste/cidade-alerta-curitiba.mp4", num_frames=30)

# for r in resultados:
#     print(
#         f"Frame {r['frame']} → "
#         f"Classe: {r['classe']} | "
#         f"Confiança: {r['confianca']:.2f}"
#     )

### Classificar full video
resultado = classificar_video_com_trechos("/home/nicolas/Videos/Treinamento-IA/teste/ric.mp4", every_n_seconds=1)

for r in resultado:
    print(f"{r['inicio']} – {r['fim']} → classe {r['classe']}")

