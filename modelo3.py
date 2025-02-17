import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt


def testa():
    # Agora o modelo já está pronto, vamos testar
    classes = ['A', 'E', 'I', 'O', 'U']

    #  0 e 5 -> A
    #  1 e 6 -> E
    #  2 e 7 -> I
    #  3 e 8 -> O 
    #  4 e 9 -> U
    for i in range(10):
        print(f"\n--------{i}:  \n" )

        image_path = "C:\\dataset\\dataset\\treino01\\letra_" + str(i) + ".jpg"
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)

        # Adicionar uma dimensão extra (batch size = 1)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Aplica o pré-processamento

        # Fazer a previsão
        resultado = model.predict(img_array)

        print("Probabilidades por classe:")
        for j, prob in enumerate(resultado[0]):
            print(f"Classe {classes[j]}: {prob:.4f}")

def grafico(history):

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Acurácia por época')
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    plt.legend(['Treino', 'Validação'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perda por época')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend(['Treino', 'Validação'], loc='upper left')
    plt.show()

# Carrega o MobileNetV2 já pré-treinado na ImageNet
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Não mudar os pesos do modelo base
base_model.trainable = False

## Preparação dos sets de treino e validação
#  Geração de imagens pro treino
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Converte [0,255] para [-1,1]
    rotation_range=30,       # Rotação aleatória
    width_shift_range=0.2,   # Deslocamento horizontal
    height_shift_range=0.2,  # Deslocamento vertical
    shear_range=0.2,         # Deformação
    zoom_range=0.2,          # Zoom
    horizontal_flip=True,    # Flip horizontal
    fill_mode='nearest'      # Preenchimento
)

#  Geração de imagens pra avaliação
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Caminhos para os datasets
dataset_train_path = os.path.abspath(r"C:\\dataset\\dataset\\treino")
dataset_val_path = os.path.abspath(r"C:\\dataset\\dataset\\validacao")

# Geração dos datasets
train_data = train_datagen.flow_from_directory(
    dataset_train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

val_data = val_datagen.flow_from_directory(
    dataset_val_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

## Criação do modelo
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),           # Reduz ao vetor de características
    Dense(128, activation='relu'),      # Camada oculta densa
    Dropout(0.5),                       # Evita overfitting
    Dense(5, activation='softmax')      # Saída para 5 gestos diferentes
])

# Compilar o modelo antes do primeiro treinamento
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy']) 

## Configurações para o treinamento com fini-tuning
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
history = model.fit(train_data, validation_data=val_data, epochs=10, callbacks=[early_stopping])

# Testa e salva o modelo sem fine-tuning
print("\nTeste do modelo sem fine-tuning\n")
testa()
grafico(history)
model.save("modelo_05.keras")
print("modelo sem fine-tuning salvo")

# A partir daqui, todos os pesos podem ser ajustados
base_model.trainable = True

# Compilar novamente o modelo para o fine-tuning
model.compile(optimizer=Adam(learning_rate=0.00001),  # Taxa de aprendizado menor para fine-tuning
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinar novamente após liberar os pesos do modelo base
checkpoint = ModelCheckpoint("modelo_melhor.keras", monitor='val_accuracy', mode='max', 
                             save_best_only=True, verbose=1)

history2 = model.fit(train_data, validation_data=val_data, epochs=5, callbacks=[checkpoint])

# Salva o modelo x
print("\nTeste do modelo com fine-tuning\n")
testa()
grafico(history2)
model.save("modelo_06.keras")