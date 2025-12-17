import cv2
import glob
import os
import numpy as np

def load_video_dataset_and_extract_frames(break_dir, merchan_dir, num_frames=6, target_size=(224, 224)):
    """
    Carrega vídeos dos diretórios e extrai frames
    
    Args:
        break_dir: diretório com vídeos de break
        merchan_dir: diretório com vídeos de merchan
        num_frames: frames por vídeo
        target_size: tamanho dos frames
    
    Returns:
        X (frames), y (labels)
    """
    
    video_extensions = ['*.mp4']
    
    frames = []
    labels = []
    
    print("Processando vídeos de break...")
    for ext in video_extensions:
        break_videos = glob.glob(os.path.join(break_dir, ext))
        for video_path in break_videos:
            print(f"Extraindo frames de: {os.path.basename(video_path)}")
            try:
                video_frames = extract_frames(video_path, num_frames, target_size)
                frames.extend(video_frames)
                labels.extend([0] * len(video_frames))
            except Exception as e:
                print(f"Erro ao processar {video_path}: {e}")
    
    print(f"\nProcessando vídeos de merchan...")
    for ext in video_extensions:
        merchan_videos = glob.glob(os.path.join(merchan_dir, ext))
        for video_path in merchan_videos:
            print(f"Extraindo frames de: {os.path.basename(video_path)}")
            try:
                video_frames = extract_frames(video_path, num_frames, target_size)
                frames.extend(video_frames)
                labels.extend([1] * len(video_frames))
            except Exception as e:
                print(f"Erro ao processar {video_path}: {e}")
    
    # Converter para numpy arrays
    X = np.array(frames, dtype=np.float32)
    y = np.array(labels)
    
    X = X / 255.0
    
    print(f"\nDataset carregado:")
    print(f"Total de frames: {len(X)}")
    print(f"Frames de break: {np.sum(y == 0)}")
    print(f"Frames de merchan: {np.sum(y == 1)}")
    print(f"Shape dos frames: {X.shape}")
    
    return X, y

def extract_frames(video_path, num_frames=6, target_size=(224, 224)):
    """
    Extrai frames uniformemente distribuídos de um vídeo
    
    Args:
        video_path: caminho para o arquivo de vídeo
        num_frames: número de frames a extrair
        target_size: tamanho de redimensionamento dos frames
    
    Returns:
        lista de frames como arrays numpy
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        print(f"Aviso: Vídeo {video_path} tem apenas {total_frames} frames")
        num_frames = total_frames
    
    # Calcular intervalos para extrair frames uniformemente
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Converter de BGR (OpenCV) para RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Redimensionar frame
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
    
    cap.release()
    
    return frames

def extract_frames_with_timestamps(video_path, every_n_seconds=1, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    frames = []
    timestamps = []

    current_time = 0.0

    while current_time < duration:
        frame_idx = int(current_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)

        frames.append(frame)
        timestamps.append(current_time)

        current_time += every_n_seconds

    cap.release()
    return frames, timestamps
