import threading
import time
import sys
from datetime import datetime
import numpy as np

import math
import jderobot
from Beacon import Beacon

from parallelIce.cameraClient import CameraClient
from parallelIce.navDataClient import NavDataClient
from parallelIce.cmdvel import CMDVel
from parallelIce.extra import Extra
from parallelIce.pose3dClient import Pose3DClient

time_cycle = 80


class MyAlgorithm(threading.Thread):

    class PID:
        def __init__(self, kp, kd, ki):
            self.kp = kp
            self.ki = ki
            self.kd = kd

        def Vel(self, error):
            return -self.kp * error[0] - self.kd * (error[0]-error[1]) - self.ki * (error[1]+error[0])



    def __init__(self, camera, navdata, pose, cmdvel, extra):
        self.camera = camera
        self.navdata = navdata
        self.pose = pose
        self.cmdvel = cmdvel
        self.extra = extra

        self.beacons = []
        self.initBeacons()
        self.minError = 0.01

        self.stop_event = threading.Event()
        self.kill_event = threading.Event()
        self.lock = threading.Lock()
        threading.Thread.__init__(self, args=self.stop_event)

    def initBeacons(self):
        self.beacons.append(Beacon('baliza1', jderobot.Pose3DData(0, 5, 0, 0, 0, 0, 0, 0), True, False))
        self.beacons.append(Beacon('baliza2', jderobot.Pose3DData(5, 0, 0, 0, 0, 0, 0, 0), False, False))
        self.beacons.append(Beacon('baliza3', jderobot.Pose3DData(0, -5, 0, 0, 0, 0, 0, 0), False, False))
        self.beacons.append(Beacon('baliza4', jderobot.Pose3DData(-5, 0, 0, 0, 0, 0, 0, 0), False, False))
        self.beacons.append(Beacon('baliza5', jderobot.Pose3DData(10, 0, 0, 0, 0, 0, 0, 0), False, False))
        self.beacons.append(Beacon('inicio', jderobot.Pose3DData(0, 0, 0, 0, 0, 0, 0, 0), False, False))

    def getNextBeacon(self):
        for beacon in self.beacons:
            if beacon.isReached() == False:
                return beacon

        return None

    def run(self):

        self.stop_event.clear()

        while (not self.kill_event.is_set()):

            start_time = datetime.now()

            if not self.stop_event.is_set():
                self.execute()

            finish_Time = datetime.now()

            dt = finish_Time - start_time
            ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
            # print (ms)
            if (ms < time_cycle):
                time.sleep((time_cycle - ms) / 1000.0)

    def stop(self):
        self.stop_event.set()

    def play(self):
        if self.is_alive():
            self.stop_event.clear()
        else:
            self.start()

    def kill(self):
        self.kill_event.set()

    PID_x = PID(5, 13, 0.2)
    PID_y = PID(5, 13, 0.2)
    error_x = [0, 0]
    error_y = [0, 0]
    vel_x = 0
    vel_y = 0
    dronx = 0
    drony = 0
    beaconx = 0
    beacony = 0

    def execute(self):
        beacon = self.getNextBeacon()
        if (beacon == None):
            print("Todas las balizas alcanzadas")
            self.cmdvel.sendCMDVel(0, 0, 0, 0, 0, 0)
            sys.exit()
        # calculo el error en cada eje
        # posicion de mi dron
        self.dronx = self.pose.getPose3D().x
        self.drony = self.pose.getPose3D().y
        # posicion de la baliza actual
        self.beaconx = beacon.getPose().x
        self.beacony = beacon.getPose().y
        # error en t actual
        self.error_x[0] = self.dronx - self.beaconx
        self.error_y[0] = self.drony - self.beacony
        # calculo las velocidades en cada eje a partir del error
        self.vel_x = self.PID_x.Vel(self.error_x)
        self.vel_y = self.PID_y.Vel(self.error_y)

        self.cmdvel.sendCMDVel(self.vel_x, self.vel_y, 0, 0, 0, 0)
        # recalculo el error para determinar si mi posicion es correcta
        self.dronx = self.pose.getPose3D().x
        self.drony = self.pose.getPose3D().y
        self.beaconx = beacon.getPose().x
        self.beacony = beacon.getPose().y
        self.error_x[0] = self.dronx-self.beaconx
        self.error_y[0] = self.drony-self.beacony
        # calculo el error total
        print("Recalculando el error...")
        print("Vel_x:", self.vel_x)
        print("Vel_y", self.vel_y)
        self.error_total = math.sqrt(math.fabs(self.error_x[0]) + math.fabs(self.error_y[0]))
        print("error total:", self.error_total)
        if (self.error_total < 0.3):
            beacon.setReached(True)
            print("Baliza conseguida!")
            # Reinicio el array de error
            self.error_x[1] = 0
            self.error_y[1] = 0
            # cambio el "tiempo" del error
        self.error_x[1] = self.error_x[0]
        self.error_y[1] = self.error_y[0]
