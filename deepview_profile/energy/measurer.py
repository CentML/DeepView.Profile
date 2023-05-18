import time
from threading import Thread
import numpy as np

import pynvml as N
from pyRAPL import Sensor

class CPUMeasurer:
    def __init__(self, interval):
        self.interval = interval
        self.power = []
        self.last_cpu = None
        self.last_dram = None

    def measurer_init(self):
        self.sensor = None
        try:
            self.sensor = Sensor()
            energy = self.sensor.energy()
            self.last_cpu = np.array(energy[0::2])
            self.last_dram = np.array(energy[1::2])
        except Exception:
            print("Warning. Failed to get CPU energy. \
                  You need to set the right permissions for pyRAPL")
            print("eg. $ sudo chmod -R a+r /sys/class/powercap/intel-rapl")

    def measurer_measure(self):
        # Get energy consumed so far (since last CPU reset)
        if self.sensor is None:
            return

        energy = self.sensor.energy()
        cpu = np.array(energy[0::2])
        dram = np.array(energy[1::2])

        # Compare against last measurement to determine energy since last measure
        diff_cpu = cpu - self.last_cpu
        dram - self.last_dram

        # 1J = 10^6 uJ
        # The cpu used this much since the last measurement
        # We have mW = 1000*J/s = 1000*(uJ/10^6)/s
        cpu_total = np.sum(diff_cpu)
        cpu_mW = 1000 * (cpu_total / 1e6) / self.interval
        self.power.append(cpu_mW)

        self.last_cpu = cpu
        self.last_dram = dram

    def measurer_deallocate(self):
        pass

    def total_energy(self):
        if len(self.power) == 0:
            return None

        # J = W * s,    1W = 1000 mW
        energy = self.interval * sum(self.power) / 1000.0
        return energy

class GPUMeasurer:
    def __init__(self, interval):
        self.interval = interval
        self.power = []

    def measurer_init(self):
        N.nvmlInit()
        self.device_handle = N.nvmlDeviceGetHandleByIndex(0)

    def measurer_measure(self):
        power = N.nvmlDeviceGetPowerUsage(self.device_handle)
        self.power.append(power)

    def measurer_deallocate(self):
        N.nvmlShutdown()

    def total_energy(self):
        # J = W * s,    1W = 1000 mW
        energy = self.interval * sum(self.power) / 1000.0
        return energy

class EnergyMeasurer:
    def __init__(self):
        self.sleep_interval = 0.1
        self.measuring = False
        self.measure_thread = None

        self.measurers = {
            "cpu": CPUMeasurer(self.sleep_interval),
            "gpu": GPUMeasurer(self.sleep_interval),
        }

    def run_measure(self):
        # Initialize
        for m in self.measurers:
            self.measurers[m].measurer_init()

        # Run measurement loop
        while self.measuring:
            for m in self.measurers:
                self.measurers[m].measurer_measure()
            time.sleep(self.sleep_interval)

        # Cleanup
        for m in self.measurers:
            self.measurers[m].measurer_deallocate()

    def begin_measurement(self):
        assert(self.measure_thread is None)
        self.measure_thread = Thread(target=self.run_measure)
        self.measuring = True
        self.measure_thread.start()

    def end_measurement(self):
        self.measuring = False
        self.measure_thread.join()
        self.measure_thread = None

    def total_energy(self):
        total_energy = 0.
        for m in self.measurers:
            e = self.measurers[m].total_energy()
            if e is not None:
                total_energy += e
        return total_energy

    def cpu_energy(self):
        return self.measurers["cpu"].total_energy()

    def gpu_energy(self):
        return self.measurers["gpu"].total_energy()
