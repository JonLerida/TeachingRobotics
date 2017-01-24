import threading
import time
from datetime import datetime
import cv2
import numpy as np

from sensors.cameraFilter import CameraFilter
from parallelIce.navDataClient import NavDataClient
from parallelIce.cmdvel import CMDVel
from parallelIce.extra import Extra
from parallelIce.pose3dClient import Pose3DClient


time_cycle = 80

class MyAlgorithm(threading.Thread):

    def __init__(self, camera, navdata, pose, cmdvel, extra):
        self.camera = camera
        self.navdata = navdata
        self.pose = pose
        self.cmdvel = cmdvel
        self.extra = extra

        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        self.lock = threading.Lock()
        threading.Thread.__init__(self, args=self.stop_event)


    def run (self):

        self.stop_event.clear()

        while (not self.kill_event.is_set()):
           
            start_time = datetime.now()

            if not self.stop_event.is_set():
                self.execute()

            finish_Time = datetime.now()

            dt = finish_Time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            #print (ms)
            if (ms < time_cycle):
                time.sleep((time_cycle - ms) / 1000.0)

    def stop (self):
        self.stop_event.set()

    def play (self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill (self):
        self.kill_event.set()

    def execute(self):
        Pi = 3.141592653589793

        # Valores de colorturner para filtrar la pelota roja
        hmin_rojo = 5.92 *180 / (2* Pi)
        hmax_rojo = 6.28 *180 / (2* Pi)
        vmin_rojo = 112.00
        vmax_rojo = 238.00
        smin_rojo = 0.85 * 255
        smax_rojo = 1.0 * 255

        filtro_min_rojo = np.array([hmin_rojo, smin_rojo, vmin_rojo])
        filtro_max_rojo = np.array([hmax_rojo, smax_rojo, vmax_rojo])
        # Valores para filtrar la pelota azul
        hmin_azul = 3.79 * 180 / (2 * Pi)
        hmax_azul = 4.19 * 180 / (2 * Pi)
        vmin_azul = 69.00
        vmax_azul = 169.00
        smin_azul = 0.54 * 255
        smax_azul = 0.8 * 255

        filtro_min_azul = np.array([hmin_azul, smin_azul, vmin_azul])
        filtro_max_azul = np.array([hmax_azul, smax_azul, vmax_azul])
        #consigo la imagen del servidor
        image_input = self.camera.getImage()
        if image_input is not None:
            #aplico filtro suavizado a la imagen
            imagen_suavizada = cv2.GaussianBlur(image_input, (5, 5), 0)
            #self.camera.setColorImage(imagen_suavizada)
            #paso la imagen de RGB a HSV
            imagen_hsv = cv2.cvtColor(imagen_suavizada, cv2.COLOR_RGB2HSV)
            #filtro la imagen y obtengo una imagen binaria
            bw_filtrada_rojo = cv2.inRange(imagen_hsv, filtro_min_rojo, filtro_max_rojo)
            bw_filtrada_azul = cv2.inRange(imagen_hsv, filtro_min_azul, filtro_max_azul)
            bw_suma= cv2.add(bw_filtrada_rojo, bw_filtrada_azul)  # 250+10 = 260 => 255

            #mejoro la imagen usando operadores morfologicos
            #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            #el resultado usando apertura no es bueno.
            #el cierre queda descartado. erosion y dilatacion(?)

            #copio la imagen binaria en una auxiliar para usarla



            copia_roja = np.copy(bw_filtrada_rojo)
            copia_azul = np.copy(bw_filtrada_azul)

            rectangle_image = np.copy(image_input)
            rectangle_image2 = np.copy(image_input)

            _,bordes_rojo,_ = cv2.findContours(copia_roja, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            _,bordes_azul,_ = cv2.findContours(copia_azul, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for cnt in bordes_rojo:
                x, y, w, h = cv2.boundingRect(cnt)
                # solo me quedo con los rectangulos que tengan cierto tamaño
                if w + h > 60 and w + h < 200:
                    rectangle_image = cv2.rectangle(rectangle_image, (x, y), (x + w, y + h), (0, 255, 20), 2)

            for cnt in bordes_azul:
                x, y, w, h = cv2.boundingRect(cnt)
                # solo me quedo con los rectangulos que tengan cierto tamaño
                if w + h > 60 and w + h < 200:
                    rectangle_image = cv2.rectangle(rectangle_image, (x, y), (x + w, y + h), (0, 0, 255), 2)


            bw_filtrada_suma = cv2.add(bw_filtrada_rojo,bw_filtrada_azul)
            self.camera.setColorImage(rectangle_image)
            self.camera.setThresoldImage(bw_filtrada_suma)

