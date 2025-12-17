import os
import sys
from sklearn.model_selection import train_test_split


sys.path.append(os.path.join(os.path.dirname(__file__), 'video-manager'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model-manager'))
from video_manager import load_video_dataset_and_extract_frames
from model_manager import create_video_cnn_model

def main():
    """
    Função principal para treinar o classificador de vídeos
    """
    
    # CONFIGURAÇÕES - AJUSTE OS CAMINHOS DOS SEUS DATASETS
    BREAK_DIR = "/home/nicolas/Zedia/Others/TensorFlow/datasets/break"      # Substitua pelo caminho real
    MERCHAN_DIR = "/home/nicolas/Zedia/Others/TensorFlow/datasets/conteudo"  # Substitua pelo caminho real
    
    NUM_FRAMES = 6
    TARGET_SIZE = (224, 224)
    
    print("=== CLASSIFICADOR DE VÍDEOS: BREAK VS MERCHAN ===\n")
    
    if not os.path.exists(BREAK_DIR):
        print(f"ERRO: Diretório {BREAK_DIR} não encontrado!")
        print("Por favor, ajuste a variável BREAK_DIR com o caminho correto.")
        return
    
    if not os.path.exists(MERCHAN_DIR):
        print(f"ERRO: Diretório {MERCHAN_DIR} não encontrado!")
        print("Por favor, ajuste a variável MERCHAN_DIR com o caminho correto.")
        return
    
    # Carregar dataset
    print("1. Carregando e processando vídeos...")
    X, y = load_video_dataset_and_extract_frames(BREAK_DIR, MERCHAN_DIR, NUM_FRAMES, TARGET_SIZE)
    
    if len(X) == 0:
        print("ERRO: Nenhum vídeo foi processado. Verifique os diretórios e formatos.")
        return
    
    # Dividir em treino e teste
    print("\n2. Dividindo dataset em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Treino: {len(X_train)} frames")
    print(f"Teste: {len(X_test)} frames")
    
    # Criar modelo
    print("\n4. Criando modelo CNN...")
    model = create_video_cnn_model(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    
    # Compilar modelo
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n5. Arquitetura do modelo:")
    model.summary()
    
    print("\n6. Iniciando treinamento...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    print("\n7. Avaliando modelo...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Accuracy no teste: {test_accuracy:.4f}")
    
    model_name= input("Digite o nome do Modelo: ")
    models_dir = os.path.join("/home/nicolas/Zedia/Others/TensorFlow/models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{model_name}.keras")
    model.save(model_path)
    print(f"\nModelo salvo em: {model_path}")

    return model, history

if __name__ == "__main__":
    main()