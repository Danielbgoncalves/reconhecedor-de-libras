# Reconhecedor de LIBRAS   ![Uma mão fechada, icon do aplicativo](icon.ico)
Este projeto utiliza uma Rede Neural Convolucional (CNN) baseada na arquitetura MobileNetV2 para reconhecer os gestos correspondentes às vogais em LIBRAS (Língua Brasileira de Sinais). O sistema é composto por módulos para treinamento do modelo, uma interface gráfica com Tkinter e uma aplicação via OpenCV para captura e teste em tempo real.


## Funcionalidades
Treinamento do Modelo:
- Uso de data augmentation para aumentar a diversidade das imagens.
- Fine-tuning do MobileNetV2 com callbacks como EarlyStopping e ModelCheckpoint.
- Geração de gráficos para monitorar a evolução da acurácia e da perda.
- Aplicação com Interface Gráfica (Tkinter):

Exibição da câmera em tempo real.
- Teste da predição do modelo diretamente pela interface.
- Tela de informações com detalhes sobre o projeto, os gestos e a importância da LIBRAS.
- Aplicação via Webcam (OpenCV):

Captura de frames em tempo real.
- Salva imagens capturadas para teste.
- Exibe as probabilidades de predição para cada classe.


## Treinamento do Modelo:

 **modelo3.py**: Script responsável pelo treinamento do modelo. Inclui:
- Preparação dos datasets de treinamento e validação com ImageDataGenerator.
- Criação e compilação do modelo baseado em MobileNetV2.
- Treinamento inicial e posterior fine-tuning.
- Funções para testar o modelo e plotar gráficos da evolução do treinamento

**app3.py**: Aplicação com interface gráfica construída com Tkinter. Realiza:

- Captura de vídeo via webcam.
- Teste da predição do modelo em tempo real.
- Exibição de informações sobre o projeto e os gestos de LIBRAS.

**camera.py**:
Script que utiliza OpenCV para capturar imagens da webcam, salvar imagens capturadas e testar as predições do modelo.
## Como Treinar

1. Clone este repositório:
   ```bash
   git clone https://github.com/Danielbgoncalves/reconhecedor-de-libras.git
    ```
2. Configure os diretórios dos datasets no script modelo3.py.
```
    dataset/
    ├── treino/
    │   ├── mao_A/
    │   │   ├── captura_01.jpg
    │   │   ├── captura_02.jpg
    │   │   └── ...
    │   ├── mao_B/
    │   └── ...
    └── validacao/
    ├── mao_A/
    ├── mao_B/
    └── ...
```
3. Instale os requisitos 
``` bash 
pip install tensorflow numpy opencv-python pillow scipy matplotlib
```

4. Execute o script para iniciar o treinamento: `python modelo3.py`
- O script realizará um primeiro treinamento sem fine-tuning, testará o modelo e exibirá gráficos da evolução.
- Em seguida, realizará o fine-tuning do modelo (liberando os pesos do MobileNetV2) e salvará o modelo final em modelo_06.keras.



## Aplicação com Interface Gráfica

Você pode usar o modelo desse repositório para as predições, basta descompactar o arquivo zipado e deixa-lo ou ainda usar um treinado por você; após treinar e salvar o modelo, execute a interface gráfica:

`python app3.py`

- A janela exibirá o vídeo da câmera e resultados da predição em tempo real.
- Utilize os botões para testar, obter informações e fechar a aplicação.
