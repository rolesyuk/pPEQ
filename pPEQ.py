#!/usr/bin/env python3

import os, sys
from os.path import basename, exists, join, splitext
import platform
import urllib.request
import progressbar
import zipfile, tarfile
from subprocess import Popen
from PyQt6 import QtGui
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QDoubleSpinBox,
    QSpinBox,
    QLabel,
    QComboBox,
    QPushButton,
    QSpacerItem,
    QCheckBox
)
import pyqtgraph as pg
from glob import glob
import numpy as np
from scipy import signal
from scipy.optimize import minimize
import soundfile as sf
import pyloudnorm as pyln
from multiprocessing import Pool
import sounddevice as sd

if getattr(sys, 'frozen', False):
    root_path = os.path.dirname(sys.executable)
else:
    root_path = os.path.dirname(os.path.realpath(__file__))
script_dir = os.path.dirname(os.path.realpath(__file__))

class ProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

def check_download_extract_abx():
    urls = dict(Linux="https://caseyconnor.org/pub/abx/ABX_lin64.tgz",
                Windows="https://caseyconnor.org/pub/abx/ABX_win64.zip",
                Darwin="https://caseyconnor.org/pub/abx/ABX_mac_arm64.zip")
    system = platform.system()
    url = urls[system]
    filename = join(root_path, basename(urls[system]))

    if not exists(filename):
        print("Downloading Lacinato ABX/Shootouter from {:s} ...".format(url))
        urllib.request.urlretrieve(url, filename, ProgressBar())

    if not exists(join(root_path, 'ABX')):
        if system in ['Windows', 'Darwin']:
            with zipfile.ZipFile(filename, 'r') as f:
                f.extractall(root_path)
        elif system in ['Linux']:
            with tarfile.open(filename, 'r:gz') as f:
                f.extractall(root_path, filter='data')

    os.environ["JAVA_HOME"] = join(root_path, 'lib', 'jre')
    if system in ['Windows']:
        abx = [join(root_path, 'ABX', 'lib', 'jre', 'bin', 'java.exe'),
               '-jar', join(root_path, 'ABX', 'lib', 'ABX.jar')]
    elif system in ['Linux', 'Darwin']:
        java = join(root_path, 'ABX', 'lib', 'jre', 'bin', 'java')
        os.chmod(java, 0o755)
        abx = [join(root_path, 'ABX', 'lib', 'jre', 'bin', 'java'),
               '-jar', join(root_path, 'ABX', 'lib', 'ABX.jar')]
    else:
        abx = None

    return abx

class FrequencyGrid():
    def __init__(self, points=200, range=[20, 20000]):
        self.f = self.generate_frequency_grid(points, range)
        self.axis = [
            (20, '20'), (30, '30'), (40, '40'), (50, '50'), (60, '60'),
            (80, '80'), (100, '100'), (150, '150'), (200, '200'),
            (300, '300'), (400, '400'), (500, '500'), (600, '600'),
            (800, '800'), (1000, '1k'), (1500, '1.5k'), (2000, '2k'),
            (3000, '3k'), (4000, '4k'), (5000, '5k'), (6000, '6k'),
            (8000, '8k'), (10000, '10k'), (15000, '15k'), (20000, '20k')]
        self.axis = [[(np.log10(x), s) for x, s in self.axis]]
        self.step = np.diff(self.f[self.f >= 100]).min()

    def generate_frequency_grid(self, points, range):
        base = np.power(range[-1] / range[0], 1 / (points - 1))
        f = base**np.arange(points)
        f *= range[0]

        return f

frequency_grid = FrequencyGrid()

class Responses():
    def __init__(self, type):
        self.responses = None

        self.read_frequency_responses(type)

    def read_frequency_responses(self, type):
        filenames = glob(join(root_path, type + 's', '*'))
        filenames.sort()
        self.responses = []
        for filename in filenames:
            ext = splitext(basename(filename))[1]
            if ext == '.csv':
                r = np.genfromtxt(filename, delimiter=',')
            elif ext == '.txt':
                r = np.genfromtxt(filename)
            else:
                r = None
            offset = np.interp(1000, r[:, 0], r[:, 1])
            r[:, 1] -= offset
            name = splitext(basename(filename))[0]
            self.responses.append({'f': r[:, 0], 'h': r[:, 1], 'name': name})
            if type == 'response':
                model = name.split(',')[0]
                self.responses[-1]['model'] = model
    
    def get_by_name(self, name):
        return [x for x in self.responses if x['name'] == name][0].copy()

class Songs():
    def __init__(self):
        filenames = glob(join(root_path, 'songs', '*'))
        filenames.sort()
        self.songs = []
        for filename in filenames:
            name = splitext(basename(filename))[0]
            self.songs.append(dict(filename=filename, name=name))

    def get_by_name(self, name):
        return [x for x in self.songs if x['name'] == name][0]

    def process_data(self, data, sos, meter):
        data = signal.sosfilt(sos, data, axis=0)
        loudness = meter.integrated_loudness(data)
        # loudness normalize audio to -16 dB LUFS
        data = pyln.normalize.loudness(data, loudness, -16.0)
        print('data.max() = {:.2f}'.format(data.max()))

        return data

    def apply_peq(self, test_cases, song_name):
        filename = self.get_by_name(song_name)['filename']
        data, sample_rate = sf.read(filename)
        for test_case in test_cases:
            test_case.current_peq.update_sample_rate(sample_rate)
        meter = pyln.Meter(sample_rate) # create BS.1770 meter

        args = []
        if len(data.shape) == 1:
            data = data[:, None]
        chs = data.shape[1]
        for test_case in test_cases:
            for ch in range(chs):
                args.append((data[:, ch], test_case.current_peq.sos, meter))
        
        with Pool(len(args)) as p:
            data = p.starmap(self.process_data, args)

        for i, test_case in enumerate(test_cases):
            track = np.vstack(data[chs*i:chs*(i+1)]).T
            sf.write(join(root_path, 'ABX', test_case.name.upper() + '.wav'),
                     track, sample_rate)

class ApoEQ():
    # According to https://www.w3.org/TR/audio-eq-cookbook/
    def __init__(self, freqs=[], gains=[], qs=[], sample_rate=44100, types='peak'):
        assert len(freqs) == len(gains)
        assert len(freqs) == len(qs)
        if isinstance(types, str):
            types = [types for i in range(len(freqs))]
        assert len(freqs) == len(types)

        self.freqs = np.array(freqs)
        self.gains = np.array(gains)
        self.qs = np.array(qs)
        self.sample_rate = sample_rate
        self.types = types
        self.sos = None
        self.response = {}
        self.base_response = None
        self.preamp = 0

        self.update_sos()

    def update_freqs_gains_qs(self, freqs, gains, qs, base_response):
        self.freqs = np.array(freqs)
        self.gains = np.array(gains)
        self.qs = np.array(qs)
        self.update_sos()
        self.update_response(base_response)

    def update_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate
        self.update_sos()

    def update_sos(self):
        soss = []
        for f, g, q, t in zip(self.freqs, self.gains, self.qs, self.types):
            A = 10**(g / 40)
            omega = 2 * np.pi * f / self.sample_rate
            alpha = np.sin(omega) / (2 * q)
            cos = np.cos(omega)
            tsAa = 2 * np.sqrt(A) * alpha

            if t == 'peak':
                sos = [1 + alpha * A,
                       -2 * cos,
                       1 - alpha * A,
                       1 + alpha / A,
                       -2 * cos,
                       1 - alpha / A]
                soss.append(sos)

            elif t == 'lowshelf':
                sos = [A * ((A + 1) - (A - 1) * cos + tsAa),
                       2 * A * ((A - 1) - (A + 1) * cos),
                       A * ((A + 1) - (A - 1) * cos - tsAa),
                       (A + 1) + (A - 1) * cos + tsAa,
                       -2 * ((A - 1) + (A + 1) * cos),
                       (A + 1) + (A - 1) * cos - tsAa]
                soss.append(sos)

            elif t == 'highshelf':
                sos = [A * ((A + 1) + (A - 1) * cos + tsAa),
                       -2 * A * ((A - 1) + (A + 1) * cos),
                       A * ((A + 1) + (A - 1) * cos - tsAa),
                       (A + 1) - (A - 1) * cos + tsAa,
                       2 * ((A - 1) - (A + 1) * cos),
                       (A + 1) - (A - 1) * cos - tsAa]
                soss.append(sos)

        sos = np.vstack(soss)
        sos /= sos[:, 3:4]
        self.sos = sos

    def update_response(self, base_response):
        self.base_response = base_response
        _, h = signal.sosfreqz(self.sos, base_response['f'], fs=self.sample_rate)
        h = 20 * np.log10(np.abs(h))
        self.preamp = -1 * h.max()
        h += base_response['h']
        offset = np.interp(1000, base_response['f'], h)
        h -= offset
        self.response['f'] = base_response['f']
        self.response['h'] = h
        self.response['name'] = base_response['name']

    def calculate_peq_response(self, fs, gs, qs, f):
        rs = np.zeros((len(fs), len(f)), dtype=np.complex128)
        for i, t in enumerate(self.types):
            s = 1j * f / fs[i]
            s2 = s**2
            A = np.sqrt(10**(gs[i] / 20))
            sqrtA = np.sqrt(A)
            if t == 'peak':
                rs[i] = (s2 + s * (A / qs[i]) + 1) / \
                        (s2 + s / (A * qs[i]) + 1)
            elif t == 'lowshelf':
                rs[i] = A * (s2 + s * (sqrtA / qs[i]) + A) / \
                            (A * s2 + s * (sqrtA / qs[i]) + 1)
            elif t == 'highshelf':
                rs[i] = A * (A * s2 + s * (sqrtA / qs[i]) + 1) / \
                            (s2 + s * (sqrtA / qs[i]) + A)

        return 20 * np.log10(np.abs(rs)).sum(axis=0)

    def calculate_peq(self, target, tol):
        nb = len(self.freqs)
        div_f = 1000
        div_g = 10
        div_q = 1
        freqs0 = [100, 200, 500, 1000, 1500, 2000, 3000, 5000, 8000, 12000]
        freqs0 = [f / div_f for f in freqs0]
        assert len(freqs0) == nb

        bounds = [(20 / div_f, 15000 / div_f)] * nb + \
                 [(-12 / div_g, 12 / div_g)] * nb + \
                 [(0.5 / div_q, 2 / div_q)] * nb + \
                 [(-12 / div_g, 12 / div_g)]

        # Fix shelf filters
        bounds[0 * nb] = (105 / div_f, 105 / div_f)
        bounds[1 * nb] = (0, 0)
        bounds[2 * nb] = (0.7 / div_q, 0.7 / div_q)
        bounds[1 * nb - 1] = (12000 / div_f, 12000 / div_f)
        bounds[2 * nb - 1] = (0, 0)
        bounds[3 * nb - 1] = (0.7 / div_q, 0.7 / div_q)

        x0 = np.hstack(freqs0 + [0.0] * nb + [0.7] * nb + [0.0])

        f = frequency_grid.f
        weights = np.ones_like(f)
        weights[f < 4000] = 10
        weights[f > 9000] = 0.1

        def put_response_on_grid(f, response):
            response_on_grid = {
                'f': f,
                'h': np.interp(f, response['f'], response['h']),
                'name': response['name']}
            return response_on_grid

        base_response = self.base_response.copy()
        target_response_on_grid = put_response_on_grid(f, target)
        base_response_on_grid = put_response_on_grid(f, self.base_response)
        # Set target > 8kHz to the headphone response
        target_response_on_grid['h'][f > 9000] = \
            base_response_on_grid['h'][f > 9000]

        def loss(x):
            self.update_freqs_gains_qs(
                div_f * x[0*nb:1*nb], div_g * x[1*nb:2*nb], div_q * x[2*nb:3*nb],
                base_response_on_grid)
            error = self.response['h'] + div_g * x[-1] - target_response_on_grid['h']
            return (error**2 * weights).sum()

        res = minimize(loss, x0, bounds=bounds, tol=tol,
                       options=dict(maxls=100, maxiter=10000, maxfun=1000000))
        print(res)
        x = res.x

        # Set high shelf to -12 by default
        x[2 * nb - 1] = -12

        self.update_freqs_gains_qs(div_f * x[0*nb:1*nb],
                                   div_g * x[1*nb:2*nb],
                                   div_q * x[2*nb:3*nb],
                                   base_response)

    def export_peq(self, name):
        print('freqs =', ['{:d}'.format(x) for x in self.freqs])
        print('gains =', ['{:.1f}'.format(x) for x in self.gains])
        print('qs =', ['{:.2f}'.format(x) for x in self.qs])

        self.export_to_equalizer_apo(name)
        self.export_to_pulse_effects(name)
        self.export_to_toneboosters(name)

        filename = join(root_path, 'peqs', name + '.csv')
        np.savetxt(filename,
                   np.stack([self.response['f'], self.response['h']]).T,
                   delimiter=',', fmt='%.1f')

    def export_to_equalizer_apo(self, name):
        eq_config = []
        eq_config.append('Preamp: {:.1f} dB'.format(self.preamp))
        types = []
        for t in self.types:
            if t == 'peak':
                types.append('PK')
            elif t == 'lowshelf':
                types.append('LSC')
            elif t == 'highshelf':
                types.append('HSC')
            else:
                raise ValueError(
                    'Only peak, lowshelf and highshelf fileters are supported')

        for i, (t, f, g, q) in enumerate(zip(types, self.freqs, self.gains, self.qs)):
            eq_config.append(
                'Filter {:d}: ON {:s} Fc {:.0f} Hz Gain {:.1f} dB Q {:.3f}'.format(
                i + 1, t, f, g, q))

        eq_config = '\n'.join(eq_config)

        filename = join(root_path, 'peqs', name + '.txt')
        with open(filename, 'w') as f:
            f.write(eq_config)

    def export_to_pulse_effects(self, name):
        with open(join(script_dir, 'pulse_effects_config_top.json'), 'r') as f:
            top_config = f.read()
        with open(join(script_dir, 'pulse_effects_config_bottom.json'), 'r') as f:
            bottom_config = f.read()

        eq_config = []
        eq_config.append(top_config)
        eq_config.append("""        "equalizer": {{
            "state": "true",
            "mode": "IIR",
            "num-bands": "{:d}",
            "input-gain": "{:.1f}",
            "output-gain": "0",
            "split-channels": "false",""".format(len(self.gains), self.preamp))

        def generate_band_config(number, type, gain, freq, q):
            if type == 'peak':
                type = 'Bell'
            elif type == 'lowshelf':
                type = 'Lo-shelf'
            elif type == 'highshelf':
                type = 'Hi-shelf'

            s = """                "band{:d}": {{
                    "type": "{:s}",
                    "mode": "APO (DR)",
                    "slope": "x1",
                    "solo": "false",
                    "mute": "false",
                    "gain": "{:.1f}",
                    "frequency": "{:d}",
                    "q": "{:.2f}"
                }},""".format(number, type, gain, freq, q)

            return s

        bands = [generate_band_config(*args) for args in zip(
            np.arange(len(self.gains)), self.types, self.gains, self.freqs, self.qs)]
        bands[-1] = bands[-1][:-1]
        eq_config.append("""            "left": {""")
        eq_config.extend(bands)
        eq_config.append("""            },
            "right": {""")
        eq_config.extend(bands)
        eq_config.append("""            }
        },""")
        eq_config.append(bottom_config)

        eq_config = '\n'.join(eq_config)

        filename = join(root_path, 'peqs', name + '.json')
        with open(filename, 'w') as f:
            f.write(eq_config)

    def export_to_toneboosters(self, name):
        # According to https://github.com/SiliconExarch/EqConverter
        eq_config = ["""<?xml version="1.0" encoding="ISO-8859-1"?><Preset>
<PresetInfo Name="{:s}" TenBand="1">""".format(name)]

        def generate_band_config(type, gain, freq, q):
            if type == 'peak':
                type = '0.21428572'
            elif type == 'lowshelf':
                type = '0.071428575'
            elif type == 'highshelf':
                type = '0.2857143'

            q_min = 0.1; q_max = 10; f_min = 16; f_max = 20000
            freq = ((freq - f_min) / (f_max - f_min))**(1 / 3)
            gain = (gain + 20) / 40
            q = ((q - q_min) / (q_max - q_min))**(1 / 3)

            s = """<Value>{:.8f}</Value>
<Value>{:.3f}</Value>
<Value>1.0</Value>
<Value>{:.8f}</Value>
<Value>{:s}</Value>
<Value>0.0</Value>""".format(freq, gain, q, type)

            return s

        bands = [generate_band_config(*args) for args in zip(
            self.types, self.gains, self.freqs, self.qs)]
        eq_config.extend(bands)

        preamp = (self.preamp + 20) / 40
        eq_config.append("""<Value>0</Value>
<Value>{:.3f}</Value>
<Value>1</Value>
<Value>0.33333334</Value>
<Value>0.05</Value>
<Value>0</Value>
</PresetInfo>
</Preset>""".format(preamp))

        eq_config = '\n'.join(eq_config)

        filename = join(root_path, 'peqs', name + '.xml')
        with open(filename, 'w') as f:
            f.write(eq_config)

class TbFilter():
    def __init__(self):
        self.f = 0.9282573 # Default to 16KHz for disabled filter
        self.gain = 0.5 # Default to 0dB for disabled filter
        self.on = False
        self.q = 0.39434525 # Default to 0.71 for disabled filter
        self.type = 0.21428572 # Analog bell

class Preferences():
    def __init__(self, name):
        # Bass, Treble, Ear Gain
        self.eq = ApoEQ(*self.get_freqs_gains_qs(0, 0, 0),
                        types=['lowshelf', 'highshelf', 'peak'])
        self.response = self.eq.response

    def get_freqs_gains_qs(self, bass, treble, ear_gain):
        return [[105, 2500, 2750], [bass, treble, ear_gain], [0.7, 0.42, 1]]

    def update(self, tilt, bass, treble, ear_gain, response):
        f = response['f']
        x = response['h']
        tilted_response = {'f': f,
                           'h': tilt * np.log2(f) + x,
                           'name': response['name']}
        self.eq.update_freqs_gains_qs(
            *self.get_freqs_gains_qs(bass, treble, ear_gain), tilted_response)

class Plot():
    def __init__(self):
        self.line = None
        self.pen = None

class TestCase():
    def __init__(self, name, main_window):
        self.main_window = main_window
        self.name = name

        self.targets = Responses('target')
        self.current_target = None
        self.preferences = Preferences(name)
        self.target_plot = Plot()
        self.target_widgets = []

        self.responses = Responses('response')
        self.current_response = None
        self.response_plot = Plot()
        self.peq_count = 10
        self.peq_labels = ['Freq', 'Gain', 'Q']
        peq_freqs = frequency_grid.f[::len(frequency_grid.f) // self.peq_count]
        if self.peq_count > 2:
            peq_types = ['lowshelf'] + ['peak'] * (self.peq_count - 2) + ['highshelf']
        else:
            peq_types = 'peak'
        self.current_peq = ApoEQ(peq_freqs,
                                 np.zeros_like(peq_freqs),
                                 np.ones_like(peq_freqs),
                                 types=peq_types)
        self.peq_plot = Plot()
        self.peq_widgets = []

        self.target_widget_labels = [
            'Target', 'Frequency Response', 'Tilt (dB/Oct)',
            'Bass (dB)', 'Treble (dB)', 'Ear Gain (dB)']
        
        self.init_gui()

    def init_gui(self):
        # Labels
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(24)
        label = QLabel(self.name.upper() + ':')
        label.setAlignment(Qt.AlignmentFlag.AlignRight |
                           Qt.AlignmentFlag.AlignVCenter)
        label.setFont(font)
        label.setFixedWidth(35)

        # Target and frequency response selection
        self.target_widgets.append(label)
        w = QComboBox()
        w.setFixedWidth(230)
        w.addItems([x['name'] for x in self.targets.responses])
        w.setCurrentText('5128 DF (HP)')
        self.current_target = self.targets.get_by_name(w.currentText())
        w.currentTextChanged.connect(self.update_and_draw_target)
        self.target_widgets.append(w)
        w = QComboBox()
        w.setFixedWidth(230)
        w.addItems([x['name'] for x in self.responses.responses])
        w.setCurrentText('Hifiman Edition XS, 5128')
        self.current_response = self.responses.get_by_name(w.currentText())
        w.currentTextChanged.connect(self.update_and_draw_response)
        self.target_widgets.append(w)
        # Tilt
        w = QDoubleSpinBox()
        w.setFixedWidth(110)
        w.setSingleStep(0.1)
        w.setMaximum(5)
        w.setMinimum(-5)
        w.setDecimals(1)
        w.setValue(-0.4)
        w.valueChanged.connect(self.update_and_draw_target)
        self.target_widgets.append(w)
        # Bass, Treble and Ear Gain
        for i in range(3):
            w = QDoubleSpinBox()
            w.setFixedWidth(110)
            w.setSingleStep(1)
            w.setMaximum(12)
            w.setMinimum(-12)
            w.setDecimals(1)
            w.valueChanged.connect(self.update_and_draw_target)
            self.target_widgets.append(w)
        self.target_widgets[-3].blockSignals(True)
        self.target_widgets[-3].setValue(6)
        self.target_widgets[-3].blockSignals(False)
        self.target_widgets[-2].blockSignals(True)
        self.target_widgets[-2].setValue(-3)
        self.target_widgets[-2].blockSignals(False)
        self.target_widgets[-1].blockSignals(True)
        self.target_widgets[-1].setValue(1)
        self.target_widgets[-1].blockSignals(False)
        # Calculate
        calculate_peq_button = QPushButton("Calculate PEQ")
        calculate_peq_button.clicked.connect(self.calculate_peq)
        self.target_widgets.append(calculate_peq_button)

        for i in range(self.peq_count):
            ws = []
            # PEQ frequency
            w = QSpinBox()
            w.setFixedWidth(110)
            w.setMinimum(20)
            w.setMaximum(15000)
            w.setSingleStep(100)
            w.setSuffix(' Hz')
            w.setValue(1000)
            w.valueChanged.connect(self.update_and_draw_peq)
            ws.append(w)
            # PEQ gain
            w = QDoubleSpinBox()
            w.setFixedWidth(100)
            w.setMinimum(-12)
            w.setMaximum(12)
            w.setSingleStep(1)
            w.setSuffix(' dB')
            w.setDecimals(1)
            w.valueChanged.connect(self.update_and_draw_peq)
            ws.append(w)
            # PEQ Q
            w = QDoubleSpinBox()
            w.setFixedWidth(70)
            w.setMinimum(0.1)
            w.setMaximum(8)
            w.setSingleStep(0.1)
            w.setValue(0.7)
            w.setDecimals(2)
            w.valueChanged.connect(self.update_and_draw_peq)
            ws.append(w)
            self.peq_widgets.append(ws)
        export_peq_button = QPushButton(text='Export PEQ')
        export_peq_button.clicked.connect(self.export_peq)
        self.peq_widgets.append([export_peq_button])

    def update_and_draw_target(self):
        tilt = self.target_widgets[3].value()
        bass = self.target_widgets[4].value()
        treble = self.target_widgets[5].value()
        ear_gain = self.target_widgets[6].value()

        target_widget = self.target_widgets[1]
        t = self.targets.get_by_name(target_widget.currentText())

        self.preferences.update(tilt, bass, treble, ear_gain, t)
        self.current_target = self.preferences.eq.response
        self.current_target['name'] = \
            self.name.upper() + ' - ' + self.current_target['name'] + \
            ' + Preferences'
        self.main_window.draw(self, 'target')

    def update_and_draw_response(self):
        response_widget = self.target_widgets[2]
        self.current_response = \
            self.responses.get_by_name(response_widget.currentText())
        self.current_response['name'] = \
            self.name.upper() + ' - ' + self.current_response['name']
        self.main_window.draw(self, 'response')
        self.update_and_draw_peq()

    def update_peq(self):
        freqs = [w[0].value() for w in self.peq_widgets[:-1]]
        gains = [w[1].value() for w in self.peq_widgets[:-1]]
        qs    = [w[2].value() for w in self.peq_widgets[:-1]]
        self.current_peq.update_freqs_gains_qs(freqs, gains, qs,
            self.current_response)

    def update_and_draw_peq(self):
        self.update_peq()
        self.current_peq.response['name'] = \
            self.current_peq.response['name'] + ' + EQ'
        self.main_window.draw(self, 'peq')

    def export_peq(self):
        self.update_peq()
        tilt     = int(np.abs(self.target_widgets[3].value()) * 10)
        bass     = int(np.abs(self.target_widgets[4].value()))
        treble   = int(np.abs(self.target_widgets[5].value()))
        ear_gain = int(np.abs(self.target_widgets[6].value()))
        self.current_peq.export_peq(self.current_response['model'] + \
            ' {:d}{:d}{:d}{:d}'.format(tilt, bass, treble, ear_gain))

    def calculate_peq(self):
        tol = float(self.main_window.tolerance.currentText().split('=')[-1])
        self.current_peq.calculate_peq(self.current_target, tol)
        for w, f in zip(self.peq_widgets[:-1], self.current_peq.freqs):
            w[0].blockSignals(True)
            w[0].setValue(int(f))
            w[0].blockSignals(False)
        for w, g in zip(self.peq_widgets[:-1], self.current_peq.gains):
            w[1].blockSignals(True)
            w[1].setValue(g)
            w[1].blockSignals(False)
        for w, q in zip(self.peq_widgets[:-1], self.current_peq.qs):
            w[2].blockSignals(True)
            w[2].setValue(q)
            w[2].blockSignals(False)

        self.update_and_draw_peq()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.test_cases = [TestCase('a', self), TestCase('b', self)]
        self.songs = Songs()
        self.songs_combobox = None
        self.plot_widget = pg.PlotWidget()
        self.abx_process = None
        self.abx = check_download_extract_abx()
        self.response_recorder = ResponseRecorder(self)
        self.tolerance = None

        self.init_gui()

    def init_gui(self):
        self.setWindowTitle("personalized PEQ")

        layout_main = QVBoxLayout()
        layout_top = QGridLayout()
        layout_top.setSpacing(5)
        layout_main.addLayout(layout_top)
        layout_bottom = QHBoxLayout()
        layout_main.addLayout(layout_bottom)
        layout_peq = QGridLayout()
        layout_peq.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout_peq.setSpacing(5)
        layout_bottom.addLayout(layout_peq)

        for i, n in enumerate(self.test_cases[0].target_widget_labels):
            layout_top.addWidget(QLabel(n), 0, i + 1)
        for i, w in enumerate(self.test_cases[0].target_widgets):
            layout_top.addWidget(w, 1, i)
        for i, w in enumerate(self.test_cases[1].target_widgets):
            layout_top.addWidget(w, 2, i)

        self.tolerance = QComboBox()
        self.tolerance.addItems(
            ['tol=1e-{:d}'.format(x) for x in range(4, 10)])
        self.tolerance.setEditable(True)
        self.tolerance.lineEdit().setReadOnly(True)
        self.tolerance.lineEdit().setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout_top.addWidget(self.tolerance, 0, i)

        layout_top.addItem(QSpacerItem(20, 1), 0, i + 1, 3, 1)

        layout_top.addWidget(QLabel('Song'), 0, i + 2)
        self.songs_combobox = QComboBox()
        self.songs_combobox.setMinimumWidth(300)
        self.songs_combobox.addItems([x['name'] for x in self.songs.songs])
        layout_top.addWidget(self.songs_combobox, 1, i + 2)

        button_run_ab = QPushButton('Run AB Test')
        layout_top.addWidget(button_run_ab, 2, i + 2)
        button_run_ab.clicked.connect(self.run_abx)

        def add_peq_widgets(test_case, layout, shift):
            layout_peq.addWidget(QLabel('Freq'), shift, 0)
            layout_peq.addWidget(QLabel('Gain'), shift, 1)
            layout_peq.addWidget(QLabel('Q'), shift, 2)
            for i, ws in enumerate(test_case.peq_widgets):
                for j, w in enumerate(ws):
                    if len(ws) == 3:
                        layout.addWidget(w, i + shift + 1, j)
                    elif len(ws) == 1:
                        layout.addWidget(w, i + shift + 1, j, 1, 3)

        layout_peq.addItem(QSpacerItem(1, 20), 0, 0, 1, 3)
        add_peq_widgets(self.test_cases[0], layout_peq, 1)

        layout_peq.addItem(QSpacerItem(1, 20),
            2 + len(self.test_cases[0].peq_widgets), 0, 1, 3)
        add_peq_widgets(self.test_cases[1], layout_peq,
                        3 + len(self.test_cases[0].peq_widgets))

        layout_peq.addItem(QSpacerItem(1, 20),
            4 + 2 * len(self.test_cases[0].peq_widgets), 0, 1, 3)
        layout_peq.addWidget(QLabel('Response Verification'),
            5 + 2 * len(self.test_cases[0].peq_widgets), 0, 1, 3)
        layout_peq.addWidget(self.response_recorder.device_label,
            6 + 2 * len(self.test_cases[0].peq_widgets), 0, 1, 1)
        layout_peq.addWidget(self.response_recorder.device_selector,
            6 + 2 * len(self.test_cases[0].peq_widgets), 1, 1, 2)
        layout_peq.addWidget(self.response_recorder.output_label,
            7 + 2 * len(self.test_cases[0].peq_widgets), 0, 1, 1)
        layout_peq.addWidget(self.response_recorder.output_selector,
            7 + 2 * len(self.test_cases[0].peq_widgets), 1, 1, 2)
        layout_peq.addWidget(self.response_recorder.record_response_button,
            8 + 2 * len(self.test_cases[0].peq_widgets), 0, 1, 3)

        layout_bottom.addWidget(self.plot_widget)

        self.setLayout(layout_main)

        pg.setConfigOptions(antialias=True)
        self.plot_widget.setBackground("w")
        styles = {"color": "black", "font-size": "18px"}
        self.plot_widget.setLabel("left", "dB", **styles)
        self.plot_widget.setLabel("bottom", "Hz", **styles)
        self.plot_widget.addLegend()
        self.plot_widget.getPlotItem().legend.setLabelTextSize('16pt')
        self.plot_widget.getPlotItem().legend.setLabelTextColor('black')
        self.plot_widget.showGrid(True, True)
        self.plot_widget.getPlotItem().getAxis('bottom').setTicks(
            frequency_grid.axis)
        self.plot_widget.getPlotItem().getAxis('left').setTextPen('black')
        self.plot_widget.getPlotItem().getAxis('bottom').setTextPen('black')
        self.plot_widget.plotItem.setLogMode(True, False)

        self.test_cases[0].target_plot.pen = \
            pg.mkPen('darkred', width=5, style=Qt.PenStyle.DashLine)
        self.test_cases[0].response_plot.pen = \
            pg.mkPen('forestgreen', width=5, style=Qt.PenStyle.SolidLine)
        self.test_cases[0].peq_plot.pen = \
            pg.mkPen('orangered', width=5, style=Qt.PenStyle.SolidLine)
        self.test_cases[1].target_plot.pen = \
            pg.mkPen('darkblue', width=5, style=Qt.PenStyle.DashLine)
        self.test_cases[1].response_plot.pen = \
            pg.mkPen('purple', width=5, style=Qt.PenStyle.SolidLine)
        self.test_cases[1].peq_plot.pen = \
            pg.mkPen('dodgerblue', width=5, style=Qt.PenStyle.SolidLine)
        self.response_recorder.plot.pen = \
            pg.mkPen('red', width=8, style=Qt.PenStyle.DotLine)

        for tc in self.test_cases:
            tc.update_and_draw_target()
            tc.update_and_draw_response()

    def plot(self, plot, x):
        if plot.line:
            self.plot_widget.removeItem(plot.line)
        plot.line = self.plot_widget.plot(x['f'], x['h'], name=x['name'],
            pen=plot.pen)
        plot.line.setAlpha(0.7, False)
        self.plot_widget.getPlotItem().setXRange(np.log10(20), np.log10(20000))
        self.plot_widget.getPlotItem().setYRange(-5, 15)

    def draw(self, test_case, type) -> None:
        if type == 'target':
            plot = test_case.target_plot
            x = test_case.current_target
        elif type == 'response':
            plot = test_case.response_plot
            x = test_case.current_response
        elif type == 'peq':
            plot = test_case.peq_plot
            x = test_case.current_peq.response

        self.plot(plot, x)

    def draw_recorded_response(self):
        plot = self.response_recorder.plot
        x = self.response_recorder.response
        self.plot(plot, x)

    def run_abx(self):
        if self.abx_process:
            self.abx_process.kill()
            self.abx_process = None

        song_name = self.songs_combobox.currentText()
        self.songs.apply_peq(self.test_cases, song_name)

        abx_config = 'filerel:{:s}.wav\nfilerel:{:s}.wav'.format(
                *[x.name.upper() for x in self.test_cases])
        with open(join(root_path, 'ABX', 'peq.abx'), 'w') as f:
            f.write(abx_config)

        if self.abx_process == None or self.abx_process.poll() != None:
            self.abx_process = Popen(self.abx)

class ResponseRecorder():
    def __init__(self, main_window):
        self.devices = [x['name'] for x in sd.query_devices()]
        self.device = None
        self.response = {
            'f': None,
            'h': None,
            'name': 'Recorded response'
        }
        self.main_window = main_window
        self.plot = Plot()

        self.output = None
        self.input = None
        self.optr = 0
        self.iptr = 0

        self.gui_init()

    def gui_init(self):
        self.device_selector = QComboBox()
        self.device_selector.addItems(self.devices)
        self.device_selector.currentTextChanged.connect(
            self.set_device)
        self.device_selector.setCurrentText('pulse')
        self.device_selector.setMaximumWidth(170)
        self.device_label = QLabel('Device')
        self.device = self.get_device_by_name(
            self.device_selector.currentText())
        self.output_selector = QCheckBox()
        self.output_selector.setChecked(True)
        self.output_selector.setMaximumWidth(170)
        self.output_label = QLabel('Test Signal')
        self.record_response_button = QPushButton('Record Response / Hide')
        self.record_response_button.clicked.connect(
            self.record)

    def get_device_by_name(self, name):
        device = [x for x in sd.query_devices() if x['name'] == name]
        if len(device) > 0:
            return device[0]
        else:
            return None

    def set_device(self):
        device_name = self.device_selector.currentText()
        self.device = self.get_device_by_name(device_name)

    def generate_chirp(self, sample_rate):
        duration = 1
        nb_samples = 2**np.ceil(np.log2(sample_rate * duration))
        t = np.arange(0, nb_samples) / sample_rate
        f_range = [0.1, 20000]
        a = np.log(f_range[-1] / f_range[0]) / t[-1]
        A = f_range[0] / a
        x = 0.8 * np.sin(2 * np.pi * A * (np.exp(a * t) - 1))

        return x

    def callback_io(self, indata, outdata, frames, time, status):
        if status:
            print('status = ', status)

        if self.optr + frames <= len(self.output):
            outdata[:] = self.output[self.optr:self.optr + frames]
            self.optr += frames
        else:
            n = len(self.output) - self.optr
            outdata[:n] = self.output[self.optr:]
            self.optr = frames - n
            outdata[n:] = self.output[:self.optr]

        if self.iptr + frames <= len(self.input):
            self.input[self.iptr:self.iptr + frames] = indata
            self.iptr += frames
        else:
            n = len(self.input) - self.iptr
            self.input[self.iptr:n] = indata[:n]
            self.iptr = frames - n
            self.input[:self.iptr] = indata[n:]

    def callback_i(self, indata, frames, time, status):
        if status:
            print('status = ', status)

        if self.iptr + frames <= len(self.input):
            self.input[self.iptr:self.iptr + frames] = indata
            self.iptr += frames
        else:
            n = len(self.input) - self.iptr
            self.input[self.iptr:n] = indata[:n]
            self.iptr = frames - n
            self.input[:self.iptr] = indata[n:]

    def record(self):
        if self.plot.line:
            self.main_window.plot_widget.removeItem(self.plot.line)
            self.plot.line = None
            return

        sample_rate = int(self.device['default_samplerate'])
        print('device =', self.device['name'])
        print('sample_rate = {:d}'.format(int(sample_rate)))
        sd.default.device = self.device['name']
        sd.default.channels = 1

        self.optr = 0 
        self.iptr = 0 

        if self.output_selector.isChecked():
            self.output = self.generate_chirp(sample_rate)[:, None]
            chirp_len = len(self.output)
            stream = sd.Stream(
                callback=self.callback_io, latency=0.1, blocksize=2048)
        else:
            chirp_len = 65536
            stream = sd.InputStream(
                callback=self.callback_i, latency=0.1, blocksize=2048)

        self.input = np.zeros((chirp_len, 1))
        stream.start()
        sd.sleep(int(2000 * chirp_len / sample_rate))
        stream.stop()
        stream.close()

        input_max = self.input.max()
        if input_max == 0:
            print('Zero input detected')
            return

        print('input.max() =', input_max)

        f_raw = np.fft.fftfreq(len(self.input), d=1/sample_rate)
        h_raw = 20 * np.log10(np.abs(np.fft.fft(self.input, axis=0))).squeeze()
        f = frequency_grid.generate_frequency_grid(500, [20, 20000])
        h_raw = h_raw[f_raw > 0]
        f_raw = f_raw[f_raw > 0]
        h = np.interp(f, f_raw, h_raw)
        h += 10 * np.log10(f)
        offset = np.interp(1000, f, h)
        h -= offset

        self.response['f'] = f
        self.response['h'] = h
        self.main_window.draw_recorded_response()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    if platform.system() == 'Darwin':
        window.showMaximized()
    else:
        window.show()
    app.exec()
    
if __name__ == '__main__':
    main()
