import threading
import time
from datetime import datetime
import cv2
import numpy as np
import math

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
    #Parámetros del filtro
    hmax = 2.82 *180 / (2* math.pi)
    hmin = 1.30*180/(2*math.pi)
    vmin = 131
    vmax = 255
    smin = 0.51*255
    smax = 1.0*255
    filtro_min = np.array([hmin, smin, vmin])
    filtro_max = np.array([hmax, smax, vmax])
    # control posicion
    error_hor = [0, 0] #Error en la imagen captada por el dron
    error_vert = [0, 0] # Error en la imagen captada por el dron


    kobuki_pose = [0, 0]
    vel_x = 0
    vel_y = 0

    class PID():
        def __init__(self, kp, kd, ki):
            self.kp = kp
            self.kd = kd
            self.ki = ki

        def Vel(self, error):
            return - self.kp * error[0] - self.kd * (error[0]-error[1]) - self.ki * (error[1]+error[0])
    PID_x = PID(0.009, 0.017, 0.0005) #con 0.000075 va bien
    PID_y = PID(0.009, 0.017, 0.0005)


    def execute(self):
        # Alto y ancho de la imagen captada por el dron (para calcular el centro)
        camera_height = self.camera.height
        camera_width = self.camera.width
        center = [camera_width/2, camera_height/2]

        #######################################
        #primera parte, procesado de la imagen#
        #######################################
        input_image = self.camera.getImage()
        if input_image is not None:

            #aplico filtro suavizado a la imagen
            imagen_suavizada = cv2.GaussianBlur(input_image, (5, 5), 0)
            #paso la imagen de RGB a HSV
            imagen_hsv = cv2.cvtColor(imagen_suavizada, cv2.COLOR_RGB2HSV)
            #filtro la imagen y obtengo una imagen binaria
            bw_filtrada = cv2.inRange(imagen_hsv, self.filtro_min, self.filtro_max)
            self.camera.setThresoldImage(bw_filtrada)

            #copio la imagen binaria en una auxiliar para usarla
            im_copia= np.copy(bw_filtrada)
            #imagen copia para dibujar el rectángulo
            rectangle_image = np.copy(input_image)
            #obtengo los bordes del objeto segmentado
            _,bordes,_ = cv2.findContours(im_copia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            #actualizo en cada pasada la posicion del kobuki
            self.kobuki_pose = [0, 0]
            for cnt in bordes:
                # w = ancho
                # h = alto
                x, y, w, h = cv2.boundingRect(cnt)
                #defino la posicion del kubuki como el centro del rectangulo
                if w > 1 and h >1:
                    self.kobuki_pose = [x+w/2, y+h/2]
                    rectangle_image = cv2.rectangle(rectangle_image, (x, y), (x + w, y + h), (255, 0, 0), 1)
            self.camera.setColorImage(rectangle_image)
        #########################################
        #segunda parte, controlador de velocidad#
        #########################################

        if self.kobuki_pose != [0, 0]:
            #error entre el kobuki y el centro de la imagen del dron
            self.error_hor[0] = center[0]-self.kobuki_pose[0]
            self.error_vert[0] = center[1]-self.kobuki_pose[1]
            self.error_total = math.sqrt(math.fabs(self.error_hor[0]) + math.fabs(self.error_vert[0]))
            if self.error_total > 5:
                #calculo la velocidad necesaria
                self.vel_x = self.PID_x.Vel(self.error_vert)
                self.vel_y = self.PID_y.Vel(self.error_hor)
                print("velocidad X: ", self.vel_x, " |Velocidad Y:  ", self.vel_y)
                self.cmdvel.sendCMDVel(-self.vel_x, -self.vel_y, 0, 0, 0, 0)
            else:
                #si el error es menor de un umbral, no cambio la velocidad
                print("Error mínimo. Mantengo posición...")
                self.cmdvel.sendCMDVel(0, 0, 0, 0, 0, 0)

            #cambio el "tiempo" del error
            self.error_hor[1] = self.error_hor[0]
            self.error_vert[1] = self.error_vert[0]
        else:
            #Si kobuki_pose = [0 0] significa que no he encontrado la tortuga
            print("Kobuki no localizado. Mantengo posición...")
            self.cmdvel.sendCMDVel(0,0, 0, 0, 0, 0)
