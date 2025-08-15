# ─────────────────────────────────────────────────────────────────────────────
# app.py — Copy Envelope (PySide6 + qdarkstyle)
# Gate drástico (sin Mix/Floor/Depth). Knobs A/R + BPM, barra con seek+volumen.
# Clic en molde = toggle + aplicar inmediato. Limpiar selección. Drag&Drop audio.
# Moldes embebidos en assets/molds (overrides opcionales en Molds/).
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import sys, json, time, threading
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import sounddevice as sd
import soundfile as sf

from PySide6.QtCore import Qt, QTimer, QSize, QEvent
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QListWidget, QListWidgetItem, QComboBox,
    QGroupBox, QAbstractItemView, QSlider, QDial, QStyle, QDoubleSpinBox
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
    level: float  # 0..1 (opcional)

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
# Motor de audio: GATE DURO
# ----------------------------
class GateFollower:
    def __init__(self, sr: int):
        self.sr = sr
        self.env = 0.0

    def process(self, target: np.ndarray, attack_ms: float, release_ms: float) -> np.ndarray:
        # ataque/relax en segundos → coeficientes (sin clicks, estable)
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
        self.playing = False
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

    def set_ar(self, attack_ms: float, release_ms: float):
        self.attack_ms = float(attack_ms); self.release_ms = float(release_ms)

    def set_molds(self, molds: List[Mold]):
        self.active_molds = list(molds)
        self.followers = [GateFollower(self.sr) for _ in self.active_molds]

    def set_play(self, flag: bool):
        self.playing = bool(flag)

    def run(self):
        while not self.stop_flag.is_set():
            if not self.playing or self.audio is None:
                time.sleep(0.01)
                continue
            chunk = self._render_chunk(1024)
            if chunk is None:
                time.sleep(0.01)
                continue
            try:
                sd.play(chunk, self.sr, blocking=False)
            except Exception:
                pass  # en CI o sin dispositivo de audio

    def _render_chunk(self, n: int):
        if self.audio is None: return None
        a = self.audio
        start = self.pos
        end = start + n
        if end > len(a):
            # loop con crossfade cortito
            head = a[start:len(a)]
            tail = a[0:(end - len(a))]
            x = np.vstack([head, tail])
            if len(x) == 0: return None
            # crossfade 10ms
            L = min(self.loop_xfade, len(tail))
            if L > 0:
                w = np.linspace(0, 1, L, dtype=np.float32)
                x[-L:, :] = (1 - w[:, None]) * x[-L:, :] + w[:, None] * a[:L, :]
            self.pos = (end - len(a))
        else:
            x = a[start:end]
            self.pos = end

        # fase en beats (para 16 beats por defecto)
        dt = n / float(self.sr)
        beats_per_sample = self.bpm / 60.0 / self.sr
        # construir máscara de gate por cada molde activo
        if self.active_molds:
            gate = np.ones(n, dtype=np.float32)
            t_beats = (np.arange(n, dtype=np.float32) + 0) * beats_per_sample
            for mold, follower in zip(self.active_molds, self.followers):
                target = np.zeros(n, dtype=np.float32)
                # a partir de group_id/length_beats, colocamos segmentos repetidos en el loop
                for seg in mold.segments:
                    # modulo a la longitud del patrón
                    # (asumimos loop de 16 beats; si el molde es más largo, recorta)
                    start_b = float(seg.start) % mold.length_beats
                    end_b   = float(seg.end)   % mold.length_beats
                    # convertimos t_beats modulo del molde
                    tb = (t_beats + self.phase_beats) % mold.length_beats
                    mask = (tb >= start_b) & (tb < end_b)
                    target[mask] = max(0.0, min(1.0, float(seg.level)))
                env = follower.process(target, self.attack_ms, self.release_ms)
                gate *= env
            x = x * gate[:, None]

        # ganancia master
        x = x * self.master_gain
        # actualizar fase (en beats)
        self.phase_beats = (self.phase_beats + n * beats_per_sample) % 16.0
        return x


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

        self.setAcceptDrops(True)  # Drag&Drop para cargar audio

        main = QVBoxLayout(self)

        # Logo centrado arriba (assets/logo.png si existe)
        self.logo_label = QLabel("")
        self.logo_label.setAlignment(Qt.AlignCenter)
        main.addWidget(self.logo_label)
        self._load_logo()

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

        top.addWidget(self.btn_load); top.addSpacing(6)
        top.addWidget(self.btn_play); top.addWidget(self.btn_stop)
        top.addStretch(1)
        main.addLayout(top)

        # Progreso / volumen
        self.lbl_time = QLabel("00:00")
        self.lbl_dur  = QLabel("00:00")
        self.sld_progress = QSlider(Qt.Horizontal); self.sld_progress.setRange(0, 0)
        self.sld_vol = QSlider(Qt.Horizontal); self.sld_vol.setRange(0, 100); self.sld_vol.setValue(100)
        prog_row = QHBoxLayout()
        prog_row.addWidget(self.lbl_time)
        prog_row.addWidget(self.sld_progress, 1)
        prog_row.addWidget(self.lbl_dur)
        prog_row.addSpacing(12)
        prog_row.addWidget(QLabel("Vol"))
        prog_row.addWidget(self.sld_vol)
        main.addLayout(prog_row)

        # Dinámica (Attack / Release knobs + BPM al lado)
        g_dyn = QGroupBox("Dinámica"); dyn = QHBoxLayout(g_dyn)

        # Knobs homogéneos: mitad de tamaño, menos sensibilidad (scale=10)
        self.kn_attack = self._make_knob(0, 300, 10, "Attack", "ms", scale=10)
        self.kn_release = self._make_knob(0, 800, 60, "Release", "ms", scale=10)
        for w in (self.kn_attack, self.kn_release):
            dyn.addWidget(w["wrap"])

        # BPM al lado (spin)
        bpm_col = QVBoxLayout()
        bpm_lbl = QLabel("BPM"); bpm_lbl.setAlignment(Qt.AlignCenter)
        self.spin_bpm = QDoubleSpinBox()
        self.spin_bpm.setRange(20, 260); self.spin_bpm.setDecimals(1); self.spin_bpm.setValue(100.0)
        self.spin_bpm.setFixedWidth(80)
        bpm_box = QWidget(); bpm_box.setLayout(QVBoxLayout()); bpm_box.layout().setContentsMargins(0,0,0,0)
        bpm_box.layout().addWidget(bpm_lbl); bpm_box.layout().addWidget(self.spin_bpm, alignment=Qt.AlignCenter)
        dyn.addWidget(bpm_box)

        main.addWidget(g_dyn)

        # Moldes (filtro por género + shot)
        g_molds = QGroupBox("Moldes"); lm = QVBoxLayout(g_molds)
        filt = QHBoxLayout()
        self.cmb_genre = QComboBox(); self.cmb_genre.addItem("Todos")
        filt.addWidget(QLabel("Género:")); filt.addWidget(self.cmb_genre)
        # Filtro adicional: Shot (family)
        self.cmb_shot = QComboBox(); self.cmb_shot.addItem("Todos")
        filt.addSpacing(12)
        filt.addWidget(QLabel("Shot:")); filt.addWidget(self.cmb_shot)
        filt.addStretch(1)
        lm.addLayout(filt)

        self.list_molds = QListWidget()
        self.list_molds.setSelectionMode(QAbstractItemView.MultiSelection)
        self.list_molds.installEventFilter(self)  # para toggle con clic simple
        lm.addWidget(self.list_molds)
        main.addWidget(g_molds)

        # Botonera inferior: Recargar + Limpiar selección
        bot = QHBoxLayout()
        self.btn_reload = QPushButton("Recargar moldes")
        self.btn_clear  = QPushButton("Limpiar selección")
        bot.addWidget(self.btn_reload)
        bot.addWidget(self.btn_clear)
        bot.addStretch(1)
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

        self.kn_attack["dial"].valueChanged.connect(self.on_ar)
        self.kn_release["dial"].valueChanged.connect(self.on_ar)

        self.cmb_genre.currentIndexChanged.connect(self.refresh_list)
        self.cmb_shot.currentIndexChanged.connect(self.refresh_list)
        self.btn_reload.clicked.connect(self.load_all_molds)
        self.btn_clear.clicked.connect(self.clear_selection)

        # Progreso / seek
        self._progress_timer = QTimer(self); self._progress_timer.timeout.connect(self._tick_progress); self._progress_timer.start(80)
        self.sld_progress.sliderPressed.connect(self._seek_press)
        self.sld_progress.sliderReleased.connect(self._seek_release)
        self._seeking = False

        # Inicial
        self.load_all_molds()

    # ---- Widgets utilitarios
    def _make_knob(self, minv, maxv, defv, title, unit, scale=1):
        wrap = QWidget(); col = QVBoxLayout(wrap); col.setContentsMargins(0,0,0,0)
        dial = QDial(); dial.setMinimum(minv); dial.setMaximum(maxv); dial.setValue(defv); dial.setNotchesVisible(True)
        lab = QLabel(f"{title}: {defv}{unit}"); lab.setAlignment(Qt.AlignCenter)
        def upd(v): lab.setText(f"{title}: {v}{unit}")
        dial.valueChanged.connect(upd)
        col.addWidget(dial); col.addWidget(lab)
        return {"wrap": wrap, "dial": dial, "label": lab}

    # ---- Logo methods
    def _load_logo(self):
        p = asset_path("logo.png")
        if p is None: 
            self.logo_label.hide()
            return
        pm = QPixmap(str(p))
        if pm.isNull():
            self.logo_label.hide()
            return
        self._logo_pix = pm
        self.logo_label.show()
        self._update_logo_pixmap()

    def _update_logo_pixmap(self):
        if hasattr(self, "_logo_pix"):
            # altura objetivo ~80 px
            h = 80
            self.logo_label.setPixmap(self._logo_pix.scaledToHeight(h, Qt.SmoothTransformation))

    def resizeEvent(self, ev):
        # mantener logo escalado
        self._update_logo_pixmap()
        return super().resizeEvent(ev)

    # ---- Carga/refresh de moldes
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
        # preservar selección por nombre si refrescamos
        selected_names = set()
        if self.list_molds.count():
            for i in range(self.list_molds.count()):
                if self.list_molds.item(i).isSelected() and i < len(self.filtered_molds):
                    selected_names.add(self.filtered_molds[i].name)

        self.all_molds.clear()
        genres = set()
        shots = set()
        files = self.find_all_mold_files()
        for p in files.values():
            try:
                m = load_mold_from_json(p)
                self.all_molds.append(m)
                genres.add(m.genre)
                shots.add(m.family)
            except Exception as e:
                print("Error leyendo", p, e)

        # actualizar combos preservando selección
        self.cmb_genre.blockSignals(True); self.cmb_shot.blockSignals(True)
        current_genre = self.cmb_genre.currentText() if self.cmb_genre.count() else "Todos"
        current_shot  = self.cmb_shot.currentText()  if hasattr(self, 'cmb_shot') and self.cmb_shot.count() else "Todos"
        self.cmb_genre.clear(); self.cmb_genre.addItem("Todos")
        for g in sorted(genres): self.cmb_genre.addItem(g)
        self.cmb_shot.clear(); self.cmb_shot.addItem("Todos")
        for s in sorted(shots): self.cmb_shot.addItem(s)
        idxg = self.cmb_genre.findText(current_genre)
        idxs = self.cmb_shot.findText(current_shot)
        self.cmb_genre.setCurrentIndex(idxg if idxg >= 0 else 0)
        self.cmb_shot.setCurrentIndex(idxs if idxs >= 0 else 0)
        self.cmb_genre.blockSignals(False); self.cmb_shot.blockSignals(False)
        self.refresh_list(preserve=selected_names)

    def refresh_list(self, preserve: set | None = None):
        genre = self.cmb_genre.currentText()
        shot  = self.cmb_shot.currentText() if hasattr(self, 'cmb_shot') else "Todos"
        self.list_molds.clear(); self.filtered_molds = []
        for m in self.all_molds:
            if genre != "Todos" and m.genre != genre: continue
            if shot  != "Todos" and m.family != shot: continue
            # Texto bonito: "ID GENRE FAMILY" -> "001 pop perc"
            item = QListWidgetItem(f"{m.group_id} {m.genre} {m.family}")
            item.setToolTip(m.name)  # nombre real del archivo
            # restaurar selección si corresponde
            if preserve and m.name in preserve:
                item.setSelected(True)
            self.list_molds.addItem(item)
            self.filtered_molds.append(m)
        self.apply_current_selection()

    # ---- Selección inmediata
    def eventFilter(self, obj, ev):
        if obj is self.list_molds and ev.type() == QEvent.MouseButtonPress:
            item = self.list_molds.itemAt(ev.pos())
            if item is not None:
                # toggle manual y aplicar en el acto
                sel = item.isSelected()
                item.setSelected(not sel)
                self.apply_current_selection()
                return True  # consumir el evento para evitar el toggle por defecto
        return super().eventFilter(obj, ev)

    def apply_current_selection(self):
        molds = []
        # mapear selección a self.filtered_molds por índice
        for i in range(self.list_molds.count()):
            if self.list_molds.item(i).isSelected():
                molds.append(self.filtered_molds[i])
        self.player.set_molds(molds)

    def clear_selection(self):
        for i in range(self.list_molds.count()):
            self.list_molds.item(i).setSelected(False)
        self.apply_current_selection()

    # ---- Slots audio
    def on_load_audio(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Cargar audio", str(Path.home()), "Audio (*.wav *.aiff *.aif)")
        if fn:
            self._load_and_play(Path(fn))

    def _load_and_play(self, path: Path):
        self.player.set_audio(path)
        dur = int(round(self.player.total_seconds))
        self.sld_progress.setRange(0, max(dur, 0))
        self.lbl_dur.setText(self._fmt_time(self.player.total_seconds))
        self.player.set_play(True)

    def on_master(self, v):
        self.player.set_master(v/100.0)

    def on_bpm(self, v):
        self.player.set_bpm(v)

    def on_ar(self, _):
        self.player.set_ar(self.kn_attack["dial"].value(), self.kn_release["dial"].value())

    # ---- Progreso
    def _tick_progress(self):
        if self.player is None or self.player.audio is None: return
        if not self._seeking:
            self.sld_progress.blockSignals(True)
            sec = int(round(self.player.pos / float(self.player.sr)))
            self.sld_progress.setValue(sec)
            self.lbl_time.setText(self._fmt_time(sec))
            self.sld_progress.blockSignals(False)

    def _seek_press(self):
        self._seeking = True

    def _seek_release(self):
        sec = self.sld_progress.value()
        self.player.pos = int(sec * self.player.sr)
        self._seeking = False

    def _fmt_time(self, secf: float | int) -> str:
        sec = int(round(secf))
        m = sec // 60
        s = sec % 60
        return f"{m:02d}:{s:02d}"

    # ---- Drag & Drop de audio
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            for url in e.mimeData().urls():
                suf = url.toLocalFile().lower()
                if suf.endswith((".wav", ".aiff", ".aif")):
                    e.acceptProposedAction()
                    return
        e.ignore()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() in (".wav", ".aiff", ".aif"):
                self._load_and_play(p)

def main():
    app = QApplication(sys.argv)
    if qdarkstyle is not None:
        try:
            app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyside6"))
        except Exception:
            pass
    w = App(); w.resize(530, 720); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()


