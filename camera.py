import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np

print("\nCarregando o modelo...")
model = load_model("modelo_06.keras")
print("Modelo carregado com sucesso!\nPrecione \n'q' para sair \n't' para testar a imagem capturada")

classes = ['A', 'E', 'I', 'O', 'U']

def salvar(imagem):
    quantidade = sum(1 for item in os.scandir('imagensTestadas') if item.is_file())
    caminho = 'imagensTestadas\\captura_' + str(quantidade) + '.jpg'
    cv2.imwrite(caminho, imagem)
    print(f"Imagem salva em: {caminho}")
    return caminho

def testar(caminho):
    
    img = image.load_img(caminho, target_size=(224, 224))
    img_array = image.img_to_array(img)

    # Adicionar uma dimensão extra (batch size = 1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Aplica o pré-processamento

    # Fazer a previsão'''
    resultado = model.predict(img_array)

    #img_array = img_array / 255.0  

    print("Probabilidades por classe:")
    for i, prob in enumerate(resultado[0]):
        print(f"Classe {classes[i]}: {prob:.4f}")

cap = cv2.VideoCapture(0) # 0 é o índice da câmera padrão

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Camera', frame) # Exibe a imagem ao vivo

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elif cv2.waitKey(1) & 0xFF == ord('t'):
        print("apertou t")
        caminho = salvar(frame)
        testar(caminho)


cap.release()
cv2.destroyAllWindows()