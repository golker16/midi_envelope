# ─────────────────────────────────────────────────────────────────────────────
# app.py — Copy Envelope (PySide6 + qdarkstyle)
# ─────────────────────────────────────────────────────────────────────────────
# Funciones clave:
# - Carga y loop de audio (WAV/AIFF), crossfade en el loop
# - Volumen master
# - BPM (manual) → evalúa segmentos en beats (lengthBeats=16)
# - Lee moldes JSON desde Molds/ con formato: genre_group_family.json
# - Filtro por género y por familia (kick/snare/hats/other)
# - Varios moldes activos a la vez (producto de ganancias)
# - Attack / Release / Depth / Floor(dB) / Mix
# - Estilo qdarkstyle, icono de ventana (assets/app.png)
# - Pie centrado: "© 2025 Gabriel Golker"
# Notas:
# - El ejecutable (PyInstaller onedir) toma el icono con assets/app.ico
# - La carpeta Molds/ se crea automáticamente junto al .exe

from __future__ import annotations
import sys
import json
import time
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QListWidget, QListWidgetItem, QSlider, QFormLayout,
    QComboBox, QDoubleSpinBox, QGroupBox, QCheckBox
)

try:
    import qdarkstyle  # Estilo oscuro para PySide6
except Exception:
    qdarkstyle = None


# ----------------------------
# Paths (compatibles con PyInstaller onedir)
# ----------------------------
APP_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
RUNTIME_DIR = Path(sys.argv[0]).resolve().parent

ASSETS_DIR_CANDIDATES = [
    RUNTIME_DIR / "assets",
    APP_DIR / "assets",
    Path(__file__).resolve().parent / "assets",
]

def asset_path(name: str) -> Path | None:
    for base in ASSETS_DIR_CANDIDATES:
        p = base / name
        if p.exists():
            return p
    return None

MOLDS_DIR = (RUNTIME_DIR / "Molds")
MOLDS_DIR.mkdir(exist_ok=True)


# ----------------------------
# Datos de molde
# ----------------------------
@dataclass
class Segment:
    start: float  # in beats
    end: float    # in beats
    level: float  # 0..1 (accent)

@dataclass
class Mold:
    name: str
    genre: str
    group_id: str
    family: str
    length_beats: int
    segments: List[Segment]

def parse_mold_filename(file: Path) -> Tuple[str, str, str]:
    # filename: genre_group_family.json
    stem = file.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return ("unknown", "000", "other")
    genre, group_id, family = parts[0], parts[1], parts[2]
    return (genre, group_id, family)

def load_mold_from_json(path: Path) -> Mold:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    length_beats = int(data.get("lengthBeats", 16))
    segs = []
    for s in data.get("segments", []):
        segs.append(Segment(
            start=float(s["start"]),
            end=float(s["end"]),
            level=float(s.get("level", 1.0))
        ))
    genre, group_id, family = parse_mold_filename(path)
    return Mold(
        name=path.name,
        genre=genre,
        group_id=group_id,
        family=family,
        length_beats=length_beats,
        segments=segs
    )


# ----------------------------
# Motor de audio
# ----------------------------
class GateFollower:
    def __init__(self, sr: int):
        self.sr = sr
        self.env = 0.0

    def process(self, target: np.ndarray, attack_ms: float, release_ms: float) -> np.ndarray:
        attack = max(attack_ms, 0.0) / 1000.0
        release = max(release_ms, 0.0) / 1000.0
        a_up = np.exp(-1.0 / (self.sr * max(attack, 1e-6)))
        a_dn = np.exp(-1.0 / (self.sr * max(release, 1e-6)))
        out = np.empty_like(target)
        env = self.env
        for i in range(len(target)):
            if target[i] > env:
                env = target[i] + (env - target[i]) * a_up
            else:
                env = target[i] + (env - target[i]) * a_dn
            out[i] = env
        self.env = float(env)
        return out

class PlayerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.audio = None        # float32 [N, C]
        self.sr = 48000
        self.pos = 0
        self.loop_xfade = int(0.01 * self.sr)  # 10ms
        self.master_gain = 1.0
        self.bpm = 100.0
        self.phase_beats = 0.0
        self.length_beats = 16.0
        self.active_molds: List[Mold] = []
        self.followers: List[GateFollower] = []
        self.attack_ms = 10.0
        self.release_ms = 60.0
        self.depth = 1.0
        self.floor_db = -24.0
        self.mix = 1.0
        self.playing = False

    # API
    def set_audio(self, path: Path):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        self.audio = data
        self.sr = sr
        self.pos = 0
        self.loop_xfade = int(0.01 * self.sr)

    def set_master(self, val: float):
        self.master_gain = float(val)

    def set_bpm(self, bpm: float):
        self.bpm = max(20.0, float(bpm))

    def set_ar(self, attack_ms: float, release_ms: float):
        self.attack_ms, self.release_ms = float(attack_ms), float(release_ms)

    def set_depth_floor_mix(self, depth: float, floor_db: float, mix: float):
        self.depth = float(np.clip(depth, 0, 1))
        self.floor_db = float(floor_db)
        self.mix = float(np.clip(mix, 0, 1))

    def set_molds(self, molds: List[Mold]):
        self.active_molds = molds
        self.followers = [GateFollower(self.sr) for _ in molds]
        self.length_beats = min([m.length_beats for m in molds]) if molds else 16.0

    def set_play(self, flag: bool):
        self.playing = bool(flag)

    # audio callback
    def _callback(self, outdata, frames, time_info, status):
        if self.audio is None or not self.playing:
            outdata[:] = 0
            return

        a = self.audio
        n = len(a)
        start = self.pos
        end = start + frames
        y = np.zeros((frames, a.shape[1]), dtype=np.float32)

        if end <= n:
            y[:] = a[start:end]
        else:
            n1 = n - start
            n2 = frames - n1
            y[:n1] = a[start:n]
            y[n1:] = a[:n2]
            xf = min(self.loop_xfade, n1, n2)
            if xf > 0:
                w = np.linspace(0, 1, xf, dtype=np.float32)
                y[n1 - xf:n1] = (1 - w)[:, None] * y[n1 - xf:n1] + w[:, None] * y[:xf]

        self.pos = (end % n)

        spb = self.sr * 60.0 / self.bpm
        inc = 1.0 / spb
        phase = (self.phase_beats + inc * np.arange(frames, dtype=np.float32)) % self.length_beats
        self.phase_beats = float((self.phase_beats + inc * frames) % self.length_beats)

        if self.active_molds:
            gains = np.ones(frames, dtype=np.float32)
            for mold, fol in zip(self.active_molds, self.followers):
                tgt = np.zeros(frames, dtype=np.float32)
                for seg in mold.segments:
                    mask = (phase >= seg.start) & (phase < seg.end)
                    lvl = float(seg.level) if seg.level is not None else 1.0
                    tgt[mask] = np.maximum(tgt[mask], lvl)
                env = fol.process(tgt, self.attack_ms, self.release_ms)
                floor_lin = 10.0 ** (self.floor_db / 20.0)
                g = floor_lin + (env ** 1.0) * (1.0 - floor_lin) * self.depth
                gains *= g.astype(np.float32)
            wet = (y * gains[:, None])
            out = (1.0 - self.mix) * y + self.mix * wet
        else:
            out = y

        out *= self.master_gain
        outdata[:] = out

    def run(self):
        block = 1024
        with sd.OutputStream(
            samplerate=self.sr,
            channels=2,
            dtype="float32",
            blocksize=block,
            callback=self._callback
        ):
            while not self.stop_flag.is_set():
                time.sleep(0.05)

    def stop(self):
        self.stop_flag.set()


# ----------------------------
# GUI
# ----------------------------
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Copy Envelope — Preview")
        # Icono de ventana (PNG)
        icon_png = asset_path("app.png")
        if icon_png is not None:
            self.setWindowIcon(QIcon(str(icon_png)))

        self.player = PlayerThread()
        self.player.start()
        self.all_molds: List[Mold] = []
        self.filtered_molds: List[Mold] = []

        main = QVBoxLayout(self)

        # Top: transporte
        top = QHBoxLayout()
        self.btn_load = QPushButton("Cargar audio…")
        self.btn_play = QPushButton("Play")
        self.btn_stop = QPushButton("Stop")
        top.addWidget(self.btn_load)
        top.addWidget(self.btn_play)
        top.addWidget(self.btn_stop)
        main.addLayout(top)

        # Global
        g_mix = QGroupBox("Global")
        f = QFormLayout(g_mix)
        self.sld_vol = QSlider(Qt.Horizontal); self.sld_vol.setRange(0, 100); self.sld_vol.setValue(80)
        self.spin_bpm = QDoubleSpinBox(); self.spin_bpm.setRange(20, 260); self.spin_bpm.setValue(100.0); self.spin_bpm.setDecimals(1)
        self.sld_mix = QSlider(Qt.Horizontal); self.sld_mix.setRange(0, 100); self.sld_mix.setValue(100)
        f.addRow("Volumen", self.sld_vol)
        f.addRow("BPM", self.spin_bpm)
        f.addRow("Mix (wet)", self.sld_mix)
        main.addWidget(g_mix)

        # Dinámica
        g_dyn = QGroupBox("Dinámica")
        fd = QFormLayout(g_dyn)
        self.sld_attack = QSlider(Qt.Horizontal); self.sld_attack.setRange(0, 300); self.sld_attack.setValue(10)
        self.sld_release = QSlider(Qt.Horizontal); self.sld_release.setRange(0, 800); self.sld_release.setValue(60)
        self.sld_depth = QSlider(Qt.Horizontal); self.sld_depth.setRange(0, 100); self.sld_depth.setValue(100)
        self.sld_floor = QSlider(Qt.Horizontal); self.sld_floor.setRange(-60, 0); self.sld_floor.setValue(-24)
        fd.addRow("Attack (ms)", self.sld_attack)
        fd.addRow("Release (ms)", self.sld_release)
        fd.addRow("Depth (%)", self.sld_depth)
        fd.addRow("Floor (dB)", self.sld_floor)
        main.addWidget(g_dyn)

        # Moldes
        g_molds = QGroupBox("Moldes")
        lm = QVBoxLayout(g_molds)
        filt = QHBoxLayout()
        self.cmb_genre = QComboBox(); self.cmb_genre.addItem("Todos")
        self.chk_kick = QCheckBox("kick"); self.chk_kick.setChecked(True)
        self.chk_snare = QCheckBox("snare/clap"); self.chk_snare.setChecked(True)
        self.chk_hats = QCheckBox("hats"); self.chk_hats.setChecked(True)
        self.chk_other = QCheckBox("other"); self.chk_other.setChecked(True)
        filt.addWidget(QLabel("Género:")); filt.addWidget(self.cmb_genre)
        filt.addStretch(1)
        filt.addWidget(self.chk_kick); filt.addWidget(self.chk_snare); filt.addWidget(self.chk_hats); filt.addWidget(self.chk_other)
        lm.addLayout(filt)
        self.list_molds = QListWidget()
        self.list_molds.setSelectionMode(self.list_molds.MultiSelection)
        lm.addWidget(self.list_molds)
        main.addWidget(g_molds)

        # Bottom buttons
        bot = QHBoxLayout()
        self.btn_reload = QPushButton("Recargar Molds/")
        self.btn_apply = QPushButton("Aplicar selección")
        bot.addWidget(self.btn_reload)
        bot.addStretch(1)
        bot.addWidget(self.btn_apply)
        main.addLayout(bot)

        # Footer centrado
        footer = QLabel("© 2025 Gabriel Golker")
        footer.setAlignment(Qt.AlignCenter)
        main.addWidget(footer)

        # Wire signals
        self.btn_load.clicked.connect(self.on_load_audio)
        self.btn_play.clicked.connect(lambda: self.player.set_play(True))
        self.btn_stop.clicked.connect(lambda: self.player.set_play(False))

        self.sld_vol.valueChanged.connect(self.on_master)
        self.spin_bpm.valueChanged.connect(self.on_bpm)
        self.sld_mix.valueChanged.connect(self.on_mix)
        self.sld_attack.valueChanged.connect(self.on_ar)
        self.sld_release.valueChanged.connect(self.on_ar)
        self.sld_depth.valueChanged.connect(self.on_depth_floor_mix)
        self.sld_floor.valueChanged.connect(self.on_depth_floor_mix)

        self.cmb_genre.currentIndexChanged.connect(self.refresh_list)
        self.chk_kick.stateChanged.connect(self.refresh_list)
        self.chk_snare.stateChanged.connect(self.refresh_list)
        self.chk_hats.stateChanged.connect(self.refresh_list)
        self.chk_other.stateChanged.connect(self.refresh_list)

        self.btn_reload.clicked.connect(self.load_all_molds)
        self.btn_apply.clicked.connect(self.apply_selected_molds)

        # Init
        self.load_all_molds()
        self.on_master(); self.on_bpm(); self.on_mix(); self.on_ar(); self.on_depth_floor_mix()

    # slots
    def on_load_audio(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Cargar audio", str(Path.home()), "Audio (*.wav *.aiff *.aif)")
        if fn:
            self.player.set_audio(Path(fn))

    def on_master(self):
        self.player.set_master(self.sld_vol.value() / 100.0)

    def on_bpm(self):
        self.player.set_bpm(self.spin_bpm.value())

    def on_mix(self):
        self.player.set_depth_floor_mix(self.sld_depth.value()/100.0, self.sld_floor.value(), self.sld_mix.value()/100.0)

    def on_ar(self):
        self.player.set_ar(self.sld_attack.value(), self.sld_release.value())

    def on_depth_floor_mix(self):
        self.player.set_depth_floor_mix(self.sld_depth.value()/100.0, self.sld_floor.value(), self.sld_mix.value()/100.0)

    def load_all_molds(self):
        self.all_molds.clear()
        genres = set()
        for p in sorted(MOLDS_DIR.glob("*.json")):
            try:
                m = load_mold_from_json(p)
                self.all_molds.append(m)
                genres.add(m.genre)
            except Exception as e:
                print("Error leyendo", p, e)

        self.cmb_genre.blockSignals(True)
        current = self.cmb_genre.currentText() if self.cmb_genre.count() else "Todos"
        self.cmb_genre.clear(); self.cmb_genre.addItem("Todos")
        for g in sorted(genres):
            self.cmb_genre.addItem(g)
        idx = self.cmb_genre.findText(current)
        self.cmb_genre.setCurrentIndex(idx if idx >= 0 else 0)
        self.cmb_genre.blockSignals(False)
        self.refresh_list()

    def refresh_list(self):
        genre = self.cmb_genre.currentText()
        fam_ok = set()
        if self.chk_kick.isChecked(): fam_ok.add("kick")
        if self.chk_snare.isChecked(): fam_ok.add("snare")
        if self.chk_hats.isChecked(): fam_ok.add("hats")
        if self.chk_other.isChecked(): fam_ok.add("other")
        self.list_molds.clear(); self.filtered_molds = []
        for m in self.all_molds:
            if genre != "Todos" and m.genre != genre:
                continue
            if m.family not in fam_ok:
                continue
            item = QListWidgetItem(f"{m.name}  [{m.genre} {m.group_id} {m.family}]")
            self.list_molds.addItem(item)
            self.filtered_molds.append(m)

    def apply_selected_molds(self):
        sel = self.list_molds.selectedIndexes()
        molds = [self.filtered_molds[i.row()] for i in sel]
        self.player.set_molds(molds)

    def closeEvent(self, e):
        self.player.stop()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    # qdarkstyle aplicado
    if qdarkstyle is not None:
        try:
            app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyside6"))
        except Exception:
            pass
    w = App()
    w.resize(900, 640)
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
