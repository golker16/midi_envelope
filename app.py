# ─────────────────────────────────────────────────────────────────────────────
# app.py — Copy Envelope (PySide6 + qdarkstyle)
# Knobs pequeños (x10 precisión), barra con seek+volumen, autoplay al cargar
# Moldes embebidos en assets/molds (overrides opcionales en Molds/)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import sys, json, time, threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import sounddevice as sd
import soundfile as sf

from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QListWidget, QListWidgetItem, QFormLayout, QComboBox,
    QDoubleSpinBox, QGroupBox, QAbstractItemView, QSlider, QDial, QStyle
)

try:
    import qdarkstyle
except Exception:
    qdarkstyle = None

# --- Paths compatibles con PyInstaller onedir
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

# Moldes embebidos y opcionales de usuario
EMBED_MOLDS_DIRS = [(base / "molds") for base in ASSETS_DIR_CANDIDATES]
RUNTIME_MOLDS_DIR = (RUNTIME_DIR / "Molds")  # overrides si existe


# ----------------------------
# Datos de molde
# ----------------------------
@dataclass
class Segment:
    start: float  # beats
    end: float    # beats
    level: float  # 0..1

@dataclass
class Mold:
    name: str
    genre: str
    group_id: str
    family: str
    length_beats: int
    segments: List[Segment]

def parse_mold_filename(file: Path) -> Tuple[str, str, str]:
    stem = file.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return ("unknown", "000", "other")
    return (parts[0], parts[1], parts[2])

def load_mold_from_json(path: Path) -> Mold:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    length_beats = int(data.get("lengthBeats", 16))
    segs = [Segment(float(s["start"]), float(s["end"]), float(s.get("level", 1.0)))
            for s in data.get("segments", [])]
    genre, group_id, family = parse_mold_filename(path)
    return Mold(path.name, genre, group_id, family, length_beats, segs)


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

        # progreso
        self.total_seconds = 0.0

    def set_audio(self, path: Path):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        self.audio = data
        self.sr = sr
        self.pos = 0
        self.loop_xfade = int(0.01 * self.sr)
        self.total_seconds = len(data) / float(sr)

    def set_master(self, val: float): self.master_gain = float(val)
    def set_bpm(self, bpm: float): self.bpm = max(20.0, float(bpm))
    def set_ar(self, attack_ms: float, release_ms: float): self.attack_ms, self.release_ms = float(attack_ms), float(release_ms)
    def set_depth_floor_mix(self, depth: float, floor_db: float, mix: float):
        self.depth = float(np.clip(depth, 0, 1))
        self.floor_db = float(floor_db)
        self.mix = float(np.clip(mix, 0, 1))
    def set_molds(self, molds: List[Mold]):
        self.active_molds = molds
        self.followers = [GateFollower(self.sr) for _ in molds]
        self.length_beats = min([m.length_beats for m in molds]) if molds else 16.0
    def set_play(self, flag: bool): self.playing = bool(flag)

    def seek_seconds(self, t_sec: float):
        if self.audio is None: return
        t_sec = max(0.0, min(float(t_sec), self.total_seconds))
        self.pos = int(t_sec * self.sr) % len(self.audio)
        spb = self.sr * 60.0 / self.bpm
        self.phase_beats = ((self.pos) / spb) % self.length_beats

    def current_seconds(self) -> float:
        if self.audio is None: return 0.0
        return self.pos / float(self.sr)

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
            samplerate=self.sr, channels=2, dtype="float32",
            blocksize=block, callback=self._callback
        ):
            while not self.stop_flag.is_set():
                time.sleep(0.05)
    def stop(self): self.stop_flag.set()


# ----------------------------
# GUI
# ----------------------------
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Copy Envelope — Preview")
        icon_png = asset_path("app.png")
        if icon_png is not None:
            self.setWindowIcon(QIcon(str(icon_png)))

        self.player = PlayerThread(); self.player.start()
        self.all_molds: List[Mold] = []; self.filtered_molds: List[Mold] = []

        main = QVBoxLayout(self)

        # Transporte (botones compactos con iconos)
        top = QHBoxLayout()
        self.btn_load = QPushButton("Cargar audio…")

        self.btn_play = QPushButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setIconSize(QSize(18, 18))
        self.btn_play.setFixedSize(28, 28)
        self.btn_play.setToolTip("Play")

        self.btn_stop = QPushButton()
        self.btn_stop.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.btn_stop.setIconSize(QSize(18, 18))
        self.btn_stop.setFixedSize(28, 28)
        self.btn_stop.setToolTip("Stop")

        top.addWidget(self.btn_load)
        top.addSpacing(8)
        top.addWidget(self.btn_play)
        top.addWidget(self.btn_stop)
        top.addStretch(1)
        main.addLayout(top)

        # Barra de reproducción + Volumen integrado
        prog_row = QHBoxLayout()
        self.lbl_time = QLabel("00:00")
        self.sld_progress = QSlider(Qt.Horizontal)
        self.sld_progress.setRange(0, 0)  # se ajusta al cargar audio
        self.sld_progress.setSingleStep(1)
        self.sld_progress.setPageStep(5)
        self.sld_progress.setTracking(False)
        self.lbl_dur = QLabel("00:00")

        self.sld_vol = QSlider(Qt.Horizontal)  # volumen junto a la barra
        self.sld_vol.setRange(0, 100); self.sld_vol.setValue(80)
        self.sld_vol.setFixedWidth(140)

        prog_row.addWidget(self.lbl_time)
        prog_row.addWidget(self.sld_progress, 1)
        prog_row.addWidget(self.lbl_dur)
        prog_row.addSpacing(12)
        prog_row.addWidget(QLabel("Vol"))
        prog_row.addWidget(self.sld_vol)
        main.addLayout(prog_row)

        # Global (BPM + Mix)
        g_mix = QGroupBox("Global"); f = QFormLayout(g_mix)
        self.spin_bpm = QDoubleSpinBox(); self.spin_bpm.setRange(20, 260); self.spin_bpm.setValue(100.0); self.spin_bpm.setDecimals(1)
        self.sld_mix = QSlider(Qt.Horizontal); self.sld_mix.setRange(0,100); self.sld_mix.setValue(100)
        f.addRow("BPM", self.spin_bpm)
        f.addRow("Mix (wet)", self.sld_mix)
        main.addWidget(g_mix)

        # Dinámica (KNOBS) — mismos, mitad de tamaño, alta resolución (×10)
        g_dyn = QGroupBox("Dinámica"); dyn = QHBoxLayout(g_dyn)
        self.kn_attack = self._make_knob(0, 300, 10, "Attack", "ms", scale=10)
        self.kn_release = self._make_knob(0, 800, 60, "Release", "ms", scale=10)
        self.kn_depth  = self._make_knob(0, 100, 100, "Depth", "%",  scale=10)
        self.kn_floor  = self._make_knob(-60, 0, -24, "Floor", "dB",  scale=10)
        for w in (self.kn_attack, self.kn_release, self.kn_depth, self.kn_floor):
            dyn.addWidget(w["wrap"])
        main.addWidget(g_dyn)

        # Moldes (solo filtro por género)
        g_molds = QGroupBox("Moldes"); lm = QVBoxLayout(g_molds)
        filt = QHBoxLayout()
        self.cmb_genre = QComboBox(); self.cmb_genre.addItem("Todos")
        filt.addWidget(QLabel("Género:")); filt.addWidget(self.cmb_genre); filt.addStretch(1)
        lm.addLayout(filt)

        self.list_molds = QListWidget()
        self.list_molds.setSelectionMode(QAbstractItemView.MultiSelection)
        lm.addWidget(self.list_molds)
        main.addWidget(g_molds)

        # Botonera inferior
        bot = QHBoxLayout()
        self.btn_reload = QPushButton("Recargar moldes")
        self.btn_apply = QPushButton("Aplicar selección")
        bot.addWidget(self.btn_reload); bot.addStretch(1); bot.addWidget(self.btn_apply)
        main.addLayout(bot)

        # Footer
        footer = QLabel("© 2025 Gabriel Golker"); footer.setAlignment(Qt.AlignCenter)
        main.addWidget(footer)

        # Señales
        self.btn_load.clicked.connect(self.on_load_audio)
        self.btn_play.clicked.connect(lambda: self.player.set_play(True))
        self.btn_stop.clicked.connect(lambda: self.player.set_play(False))

        self.sld_vol.valueChanged.connect(self.on_master)
        self.spin_bpm.valueChanged.connect(self.on_bpm)
        self.sld_mix.valueChanged.connect(self.on_mix)

        self.kn_attack["dial"].valueChanged.connect(self.on_ar)
        self.kn_release["dial"].valueChanged.connect(self.on_ar)
        self.kn_depth["dial"].valueChanged.connect(self.on_depth_floor_mix)
        self.kn_floor["dial"].valueChanged.connect(self.on_depth_floor_mix)

        self.cmb_genre.currentIndexChanged.connect(self.refresh_list)
        self.btn_reload.clicked.connect(self.load_all_molds)
        self.btn_apply.clicked.connect(self.apply_selected_molds)

        # Progreso / seek
        self.sld_progress.sliderPressed.connect(self._pause_ui_seek)
        self.sld_progress.sliderReleased.connect(self._apply_seek)
        self.sld_progress.valueChanged.connect(self._preview_time_label)

        # Timer para refrescar barra de progreso
        self.timer = QTimer(self)
        self.timer.setInterval(50)  # 20 fps
        self.timer.timeout.connect(self._tick_progress)
        self.timer.start()

        # Init
        self.load_all_molds()
        self.on_master(); self.on_bpm(); self.on_mix(); self.on_ar(); self.on_depth_floor_mix()
        self._user_seeking = False

    # ---- helpers UI
    def _make_knob(self, minv:int, maxv:int, init:float, label:str, unit:str, scale:int=10):
        """
        Knob homogéneo:
          - Tamaño 40x40 (mitad de antes)
          - Escala 'scale' -> más resolución (menos sensibilidad)
          - Muestra valor en tiempo real debajo
        """
        dial = QDial()
        dial.setNotchesVisible(True)
        dial.setWrapping(False)
        dial.setFixedSize(40, 40)  # mitad de tamaño
        # mapping int <-> valor real
        rng = (maxv - minv)
        dial.setRange(0, int(rng * scale))
        dial.setSingleStep(1)
        dial.setValue(int((init - minv) * scale))

        title = QLabel(label); title.setAlignment(Qt.AlignCenter)
        val_lbl = QLabel("")   ; val_lbl.setAlignment(Qt.AlignCenter)

        # actualiza etiqueta con valor real
        def update_label():
            val = minv + dial.value() / float(scale)
            if unit == "%":
                val_lbl.setText(f"{val:0.1f} {unit}")
            elif unit == "ms":
                val_lbl.setText(f"{val:0.1f} {unit}")
            elif unit == "dB":
                val_lbl.setText(f"{val:0.1f} {unit}")
            else:
                val_lbl.setText(f"{val:0.2f} {unit}")
        dial.valueChanged.connect(update_label)
        update_label()

        box = QWidget()
        lay = QVBoxLayout(box); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(title); lay.addWidget(dial, alignment=Qt.AlignCenter); lay.addWidget(val_lbl)

        return {"wrap": box, "dial": dial, "min": minv, "max": maxv, "scale": scale, "val_lbl": val_lbl}

    # conversion helpers
    def _kn_val(self, k) -> float:
        return k["min"] + k["dial"].value() / float(k["scale"])

    # ---- Carga de moldes (embebidos + overrides)
    def find_all_mold_files(self) -> Dict[str, Path]:
        files: Dict[str, Path] = {}
        for d in EMBED_MOLDS_DIRS:
            if d.exists():
                for p in sorted(d.glob("*.json")):
                    files.setdefault(p.name, p)
        if RUNTIME_MOLDS_DIR.exists():
            for p in sorted(RUNTIME_MOLDS_DIR.glob("*.json")):
                files[p.name] = p
        return files

    def load_all_molds(self):
        self.all_molds.clear()
        genres = set()
        files = self.find_all_mold_files()
        for p in files.values():
            try:
                m = load_mold_from_json(p)
                self.all_molds.append(m)
                genres.add(m.genre)
            except Exception as e:
                print("Error leyendo", p, e)

        self.cmb_genre.blockSignals(True)
        current = self.cmb_genre.currentText() if self.cmb_genre.count() else "Todos"
        self.cmb_genre.clear(); self.cmb_genre.addItem("Todos")
        for g in sorted(genres): self.cmb_genre.addItem(g)
        idx = self.cmb_genre.findText(current)
        self.cmb_genre.setCurrentIndex(idx if idx >= 0 else 0)
        self.cmb_genre.blockSignals(False)
        self.refresh_list()

    def refresh_list(self):
        genre = self.cmb_genre.currentText()
        self.list_molds.clear(); self.filtered_molds = []
        for m in self.all_molds:
            if genre != "Todos" and m.genre != genre: continue
            item = QListWidgetItem(f"{m.name}  [{m.genre} {m.group_id} {m.family}]")
            self.list_molds.addItem(item)
            self.filtered_molds.append(m)

    # ---- Slots
    def on_load_audio(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Cargar audio", str(Path.home()), "Audio (*.wav *.aiff *.aif)")
        if fn:
            self.player.set_audio(Path(fn))
            # rango de progreso
            dur = int(round(self.player.total_seconds))
            self.sld_progress.setRange(0, max(dur, 0))
            self.lbl_dur.setText(self._fmt_time(self.player.total_seconds))
            # autoplay
            self.player.set_play(True)

    def on_master(self):
        self.player.set_master(self.sld_vol.value()/100.0)

    def on_bpm(self): self.player.set_bpm(self.spin_bpm.value())
    def on_mix(self):
        depth_pct = self._kn_val(self.kn_depth)   # 0..100
        floor_db = self._kn_val(self.kn_floor)    # -60..0
        self.player.set_depth_floor_mix(depth_pct/100.0, floor_db, self.sld_mix.value()/100.0)

    def on_ar(self):
        self.player.set_ar(self._kn_val(self.kn_attack), self._kn_val(self.kn_release))

    def on_depth_floor_mix(self): self.on_mix()

    def apply_selected_molds(self):
        sel = self.list_molds.selectedIndexes()
        molds = [self.filtered_molds[i.row()] for i in sel]
        self.player.set_molds(molds)

    # ---- Progreso / Seek
    def _tick_progress(self):
        if getattr(self, "_user_seeking", False):
            return
        sec = self.player.current_seconds()
        self.lbl_time.setText(self._fmt_time(sec))
        if self.sld_progress.maximum() > 0:
            self.sld_progress.blockSignals(True)
            self.sld_progress.setValue(int(sec))
            self.sld_progress.blockSignals(False)

    def _pause_ui_seek(self): self._user_seeking = True
    def _apply_seek(self):
        val = self.sld_progress.value()
        self.player.seek_seconds(val)
        self._user_seeking = False
    def _preview_time_label(self):
        if getattr(self, "_user_seeking", False):
            self.lbl_time.setText(self._fmt_time(self.sld_progress.value()))

    def _fmt_time(self, sec: float) -> str:
        sec = int(sec)
        m = sec // 60
        s = sec % 60
        return f"{m:02d}:{s:02d}"

    def closeEvent(self, e):
        self.player.stop()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    if qdarkstyle is not None:
        try:
            app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyside6"))
        except Exception:
            pass
    w = App(); w.resize(980, 720); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

