# Classificador de Vídeos: Break vs Merchan

Um classificador de vídeos usando CNN (Rede Neural Convolucional) para distinguir entre vídeos de **break**, **merchan** e **conteudo**.

## 📋 Como Funciona

1. Extrai 6 frames de cada vídeo
2. Treina uma CNN para classificar os frames
3. Salva o modelo treinado

## 📁 Estrutura de Pastas

```
datasets/
├── merchan/          # Vídeos de merchan
│   ├── video1.mp4
│   └── video2.mp4
└── conteudo/       # Vídeos de conteudo
    ├── video1.mp4
    └── video2.mp4
```

## 🚀 Como Usar

1. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

2. **Colocar seus vídeos nas pastas corretas**

3. **Executar:**
```bash
python main-trainer.py
```

4. **Digite um nome para salvar o modelo**

## 📦 Dependências

- TensorFlow
- OpenCV
- NumPy  
- scikit-learn

## 📊 Saída

- Modelo treinado salvo em `models/`
- Relatório de accuracy no console
- Histórico de treinamento

## 🎯 Formatos Suportados

- `.mp4`