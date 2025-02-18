print("Iniciando o carregamento dos módulos...")
print("Mensagens abaixo informam que a CPU está sendo otimizada para o processamento, você pode ignorar.")
print("Pode demorar alguns segundos...")

import tkinter as tk
from tkinter import messagebox
import cv2
import os
import sys
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

# Carrega o modelo
print("\nCarregando o modelo...\n\n")
# Caminho para o modelo
if getattr(sys, 'frozen', False):
    # Se rodando como executável
    base_path = sys._MEIPASS
else:
    # Se rodando como script normal
    base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_path, "modelo_06.keras")
model = load_model(model_path)
print("\n\nModelo carregado com sucesso!")


#
#
#
#   Falta adicionar icon -> ok
#   certificar de q as imagens de  exmplos estao sendo carregdas -> ok
#   e tirar o terminal -> ok
#
#
#
#

classes = ['A', 'E', 'I', 'O', 'U']

def testar_frame(frame):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (224, 224))
        img_array = np.expand_dims(resized.astype("float32"), axis=0)
        img_array = preprocess_input(img_array)
        resultado = model.predict(img_array)
        return resultado[0]
    except Exception as e:
        messagebox.showerror("Erro", f"Ocorreu um erro ao testar a imagem tente fechar e abrir novamente:\n{e}")

class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Identificador de Vogais em LIBRAS")
        self.window.configure(bg="#03060d")
        self.window.geometry("800x600")
        self.window.protocol('WM_DELETE_WINDOW', self.on_close)
        
        # Cria os frames para a tela da câmera e a tela de informações
        self.camera_frame = tk.Frame(window, bg="#03060d")
        self.info_frame = tk.Frame(window, bg="#03060d")
        
        # ======= Tela da Câmera =======
        # Label para exibição do vídeo
        self.video_label = tk.Label(self.camera_frame, bg="#03060d")
        self.video_label.pack(pady=10)
        
        # Label para exibição dos resultados
        self.result_label = tk.Label(
            self.camera_frame,
            text="Resultado da predição:",
            font=("Helvetica", 15, "bold"),
            bg="#03060d",
            fg="white",
            justify="left"
        )
        self.result_label.pack(pady=10)
        
        # Frame para os botões da tela da câmera
        self.btn_frame = tk.Frame(self.camera_frame, bg="#03060d")
        self.btn_frame.pack(pady=10)
        
        # Botão para testar a imagem
        self.btn_test = tk.Button(
            self.btn_frame,
            text="Testar",
            command=self.testar_imagem,
            bg="#4CAF50",
            fg="white",
            font=("Helvetica", 14, "bold"),
            padx=10,
            pady=5,
            bd=0
        )
        self.btn_test.pack(side=tk.LEFT, padx=10)
        
        # Botão para abrir a tela de informações
        self.btn_info = tk.Button(
            self.btn_frame,
            text="Info",
            command=self.show_info,
            bg="#3498DB",
            fg="white",
            font=("Helvetica", 14, "bold"),
            padx=10,
            pady=5,
            bd=0
        )
        self.btn_info.pack(side=tk.LEFT, padx=10)
        
        # Botão para fechar a aplicação
        self.btn_quit = tk.Button(
            self.btn_frame,
            text="Fechar",
            command=self.on_close,
            bg="#f44336",
            fg="white",
            font=("Helvetica", 14, "bold"),
            padx=10,
            pady=5,
            bd=0
        )
        self.btn_quit.pack(side=tk.LEFT, padx=10)
        
        # ======= Tela de Informações =======
        self.info_title = tk.Label(
            self.info_frame,
            text="Sobre a Aplicação",
            font=("Helvetica", 15),
            bg="#03060d",
            fg="white"
        )
        self.info_title.pack(pady=(20,10))
        
        # Frame para as imagens dos gestos
        self.gestos_frame = tk.Frame(self.info_frame, bg="#03060d")
        self.gestos_frame.pack(pady=10)
        
        # Criar caminho absoluto para a pasta "gestos"
        gestos_path = os.path.join(base_path, "gestos")

        # Lista de arquivos na pasta "gestos"
        gesto_filenames = [
            os.path.join(gestos_path, "A.png"),
            os.path.join(gestos_path, "E.png"),
            os.path.join(gestos_path, "I.png"),
            os.path.join(gestos_path, "O.png"),
            os.path.join(gestos_path, "U.png"),
        ]

        # Armazena as referências para evitar que o garbage collector limpe as imagens
        self.gesto_images = []
        for filename in gesto_filenames:
            try:
                img = Image.open(filename)
                img = img.resize((100, 100))  # Ajuste o tamanho conforme necessário
                photo = ImageTk.PhotoImage(img)
                self.gesto_images.append(photo)
                lbl = tk.Label(self.gestos_frame, image=photo, bg="#03060d")
                lbl.pack(side=tk.LEFT, padx=5)
            except Exception as e:
                print(f"Erro ao carregar imagem {filename}: {e}")
        
        # Label informativo
        info_text = (
            '''
            Esta aplicação utiliza uma Rede Neural Convolucional (CNN) para 
            identificar as vogais em LIBRAS. Ela foi treinada com 5000
            imagens de mãos realizando os sinais das vogais A, E, I, O e U. Em 
            situações de gestos incorretos a precisão pode ser afetada, então 
            busque usar essa aplicação pro bem: aprender os gestos desse idioma
            oficial do Brasil! 
            Obs: A rede, esporadicamente, pode errar a classificações 
            (principalmente entre I e U).
            
            Língua Brasileira de Sinais (Libras):
            A Libras é a língua de sinais do Brasil, essencial para a inclusão social 
            da comunidade surda. Aprender Libras promove acessibilidade, respeito 
            à cultura surda e igualdade de direitos, fortalecendo a comunicação e a 
            cidadania.

            Programador: Daniel Borges Gonçalves
            No GitHub: https://github.com/Danielbgoncalves'''
        )
        
        self.info_label = tk.Label(
            self.info_frame,
            text=info_text,
            font=("Helvetica", 16),
            bg="#03060d",
            fg="white",
            justify="left"
        )
        self.info_label.pack(padx=20, pady=10)
        
        # Botão para voltar à tela da câmera
        self.btn_back = tk.Button(
            self.info_frame,
            text="Voltar",
            command=self.show_camera,
            bg="#3498DB",
            fg="white",
            font=("Helvetica", 14, "bold"),
            padx=10,
            pady=5,
            bd=0
        )
        self.btn_back.pack(pady=20)
        
        # Inicia exibindo a tela da câmera
        self.camera_frame.pack(fill="both", expand=True)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            #messagebox.showerror("Erro", "Não foi possível acessar a câmera!\nTalvez ela esteja bloqueada ou não exista.")
            self.video_label.config(text="\nErro: Não foi possível acessar a câmera!\nTalvez ela esteja bloqueada ou não exista.\n", 
                                    bg="white", font=("Helvetica", 14))
            self.in_camera = False
        else:
            self.in_camera = True
            self.update_video()

    def update_video(self):
        if self.in_camera:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
            self.window.after(10, self.update_video)
      
    
    def testar_imagem(self):
        if self.current_frame is not None:
            resultado = testar_frame(self.current_frame)
            texto = "Probabilidades por classe:\n"
            for i, prob in enumerate(resultado):
                prob = round(prob, 4) * 100
                texto += f"Classe {classes[i]}..........: {prob:.2f}%\n"
            self.result_label.config(text=texto)
        else:
            self.result_label.config(text="Nenhuma imagem capturada!")
    
    def show_info(self):
        # Para a atualização da câmera e libera o dispositivo
        self.in_camera = False
        if self.cap:
            self.cap.release()
            self.cap = None
        # Oculta a tela da câmera e exibe a tela de informações
        self.camera_frame.pack_forget()
        self.info_frame.pack(fill="both", expand=True)
    
    def show_camera(self):
        # Oculta a tela de informações, reinicia a captura e volta à tela da câmera
        self.info_frame.pack_forget()
        self.cap = cv2.VideoCapture(0)
        self.in_camera = True
        self.camera_frame.pack(fill="both", expand=True)
        self.update_video()
    
    def on_close(self):
        if self.cap:
            self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.minsize(760, 760)
    root.mainloop()
