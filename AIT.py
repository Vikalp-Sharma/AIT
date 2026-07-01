#!/usr/bin/env python3
"""
AIT — AI Trainer & Tester
==========================
Python 3.12 · sklearn · sounddevice · matplotlib · tkinter

Single-file AI training and testing tool.
  • Add word classes (name + ID)
  • Hold R to record 1-second samples from your PC mic
  • Train a sklearn MLPClassifier on those samples
  • Live-test: hold R, see which word the AI predicts + confidence bar

No hardware required beyond a PC microphone.

Install:
    pip install numpy scikit-learn joblib sounddevice matplotlib
Run:
    python AIT.py
"""

import json
import math
import os
import queue
import sys
import threading
import time
import traceback
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import sounddevice as sd
    _HAS_AUDIO = True
except ImportError:
    _HAS_AUDIO = False

# ══════════════════════════════════════════════════════════════
#  Paths & constants
# ══════════════════════════════════════════════════════════════
ROOT         = Path(__file__).resolve().parent
DATA_DIR     = ROOT / "recordings"
MODEL_PKL    = ROOT / "model.pkl"
META_FILE    = ROOT / "meta.json"
WEIGHTS_JSON = ROOT / "weights.json"

SR        = 16_000
DURATION  = 1.0
N_FFT     = 256
HOP       = 128
N_MELS    = 26
N_MFCC    = 13
INPUT_DIM = N_MFCC * 2
HIDDEN    = (64, 32)

BGN_CID  = 0
BGN_NAME = "BGN"

# ══════════════════════════════════════════════════════════════
#  MFCC feature extraction (pure numpy)
# ══════════════════════════════════════════════════════════════

def _mel_fb():
    nb   = N_FFT // 2 + 1
    m_lo = 2595.0 * math.log10(1.0 + 0.0      / 700.0)
    m_hi = 2595.0 * math.log10(1.0 + SR / 2.0 / 700.0)
    pts  = np.linspace(m_lo, m_hi, N_MELS + 2)
    hz   = 700.0 * (10.0 ** (pts / 2595.0) - 1.0)
    bins = np.floor((N_FFT + 1) * hz / SR).astype(int)
    fb   = np.zeros((N_MELS, nb), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        lo, c, hi = bins[m-1], bins[m], bins[m+1]
        for k in range(lo, c): fb[m-1,k] = (k-lo)/(c-lo) if c!=lo else 0.0
        for k in range(c, hi): fb[m-1,k] = (hi-k)/(hi-c) if hi!=c else 0.0
    return fb

def _dct_mat():
    d = np.zeros((N_MFCC, N_MELS), dtype=np.float32)
    for i in range(N_MFCC):
        for j in range(N_MELS):
            d[i,j] = math.cos(math.pi * i * (j + 0.5) / N_MELS)
    d[0] *= math.sqrt(1.0 / N_MELS)
    for i in range(1, N_MFCC): d[i] *= math.sqrt(2.0 / N_MELS)
    return d

_HAMMING = np.array([0.54 - 0.46*math.cos(2*math.pi*n/(N_FFT-1))
                     for n in range(N_FFT)], dtype=np.float32)
_MEL_FB  = _mel_fb()
_DCT_MAT = _dct_mat()

def extract_features(audio: np.ndarray) -> np.ndarray:
    nf  = max(1, (len(audio) - N_FFT) // HOP + 1)
    acc = np.zeros((N_MFCC, nf), dtype=np.float32)
    for f in range(nf):
        seg = audio[f*HOP : f*HOP+N_FFT]
        if len(seg) < N_FFT: seg = np.pad(seg, (0, N_FFT-len(seg)))
        sp  = np.fft.rfft(seg.astype(np.float32) * _HAMMING)
        pwr = (np.abs(sp) ** 2) / N_FFT
        acc[:,f] = _DCT_MAT @ np.log(np.maximum(_MEL_FB @ pwr, 1e-10))
    return np.concatenate([acc.mean(1), acc.std(1)]).astype(np.float32)

# ══════════════════════════════════════════════════════════════
#  Colour palette
# ══════════════════════════════════════════════════════════════
BG      = "#0e1117"
SURFACE = "#1a1d27"
CARD    = "#22263a"
BORDER  = "#2e3452"
ACCENT  = "#4f8ef7"
GREEN   = "#3ecf8e"
RED     = "#f87171"
AMBER   = "#fbbf24"
TEXT    = "#e2e8f0"
MUTED   = "#64748b"

PAL = ["#4f8ef7","#3ecf8e","#f87171","#fbbf24","#a78bfa",
       "#f472b6","#34d399","#fb923c","#38bdf8","#a3e635",
       "#c084fc","#fb7185","#4ade80","#facc15","#60a5fa",
       "#f97316","#22d3ee","#84cc16","#e879f9","#f43f5e"]

# ══════════════════════════════════════════════════════════════
#  Main App
# ══════════════════════════════════════════════════════════════

class AIT:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("AIT — AI Trainer & Tester")
        root.configure(bg=BG)
        root.geometry("900x680")
        root.minsize(780, 560)

        # State
        self.classes:    dict[int, str]  = {}
        self.recordings: dict[int, list] = {}
        self.pipeline:   Pipeline | None = None
        self.cid_list:   list[int]       = []
        self.current_id: int | None      = None
        self.recording:  bool            = False
        self.r_held:     bool            = False
        self.training:   bool            = False
        self.mode:       str             = "train"   # "train" | "test"

        self.q: queue.Queue = queue.Queue()

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load_meta()
        self._build_ui()
        self._bind_keys()
        self._poll()

        if not _HAS_AUDIO:
            messagebox.showwarning("No Audio",
                "sounddevice not found.\nInstall it:  pip install sounddevice")

    # ── Persistence ───────────────────────────────────────────

    def _load_meta(self):
        if not META_FILE.exists(): return
        try:
            meta = json.loads(META_FILE.read_text())
            self.classes = {int(k): v for k,v in meta.get("classes",{}).items()}
            for cid in self.classes:
                self.recordings[cid] = []
                d = DATA_DIR / str(cid)
                if d.is_dir():
                    for fp in sorted(d.glob("*.npy")):
                        try: self.recordings[cid].append(np.load(fp))
                        except: pass
        except: pass
        # Load model if available
        if MODEL_PKL.exists():
            try:
                self.pipeline = joblib.load(MODEL_PKL)
                if WEIGHTS_JSON.exists():
                    w = json.loads(WEIGHTS_JSON.read_text())
                    self.cid_list = [int(c) for c in w.get("cid_list",[])]
                    self.classes.update({int(k):v for k,v in w.get("classes",{}).items()})
            except: pass

    def _save_meta(self):
        META_FILE.write_text(
            json.dumps({"classes":{str(k):v for k,v in self.classes.items()}},
                       indent=2))

    # ── Queue poll ────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                tag, val = self.q.get_nowait()
                if tag == "rec_prog":
                    self.prog_rec.config(value=float(val))
                elif tag == "rec_done":
                    self._rec_done_ui()
                elif tag == "train_msg":
                    self.lbl_status.config(text=val, fg=ACCENT)
                elif tag == "train_done":
                    self._train_done_ui(val)
                elif tag == "train_err":
                    self._train_err_ui(val)
                elif tag == "infer":
                    probs, cid_list = val
                    self._show_prediction(probs, cid_list)
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    # ══════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── Header bar ────────────────────────────────────────
        hdr = tk.Frame(self.root, bg="#0a0d15", height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="AIT", bg="#0a0d15", fg=ACCENT,
                 font=("Courier", 22, "bold")).pack(side="left", padx=18, pady=8)
        tk.Label(hdr, text="AI Trainer & Tester", bg="#0a0d15", fg=MUTED,
                 font=("Courier", 10)).pack(side="left")

        # Mode toggle
        tog = tk.Frame(hdr, bg="#0a0d15")
        tog.pack(side="right", padx=16)

        self.btn_train_mode = tk.Button(
            tog, text="TRAIN", command=lambda: self._set_mode("train"),
            bg=ACCENT, fg="white", relief="flat",
            font=("Courier", 9, "bold"), padx=14, pady=6, cursor="hand2")
        self.btn_train_mode.pack(side="left", padx=(0,2))

        self.btn_test_mode = tk.Button(
            tog, text="TEST", command=lambda: self._set_mode("test"),
            bg=CARD, fg=MUTED, relief="flat",
            font=("Courier", 9, "bold"), padx=14, pady=6, cursor="hand2")
        self.btn_test_mode.pack(side="left")

        # ── Body ──────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        # Left panel
        self.left = tk.Frame(body, bg=BG, width=300)
        self.left.pack(side="left", fill="y", padx=(14,0), pady=12)
        self.left.pack_propagate(False)
        self._build_left()

        # Right panel
        self.right = tk.Frame(body, bg=BG)
        self.right.pack(side="left", fill="both", expand=True, padx=12, pady=12)

        self.train_panel = tk.Frame(self.right, bg=BG)
        self.test_panel  = tk.Frame(self.right, bg=BG)

        self._build_train_panel()
        self._build_test_panel()

        self._set_mode("train")

    # ── Left sidebar: class list ───────────────────────────────

    def _build_left(self):
        p = self.left

        # Add class section
        sec = tk.Frame(p, bg=SURFACE, bd=0)
        sec.pack(fill="x", pady=(0,8))

        tk.Label(sec, text="ADD CLASS", bg=SURFACE, fg=MUTED,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=12, pady=(10,4))

        row = tk.Frame(sec, bg=SURFACE)
        row.pack(fill="x", padx=10, pady=(0,6))

        tk.Label(row, text="Name", bg=SURFACE, fg=MUTED,
                 font=("Courier", 8)).grid(row=0, column=0, sticky="w")
        self.ent_name = tk.Entry(row, bg=CARD, fg=TEXT, insertbackground=TEXT,
                                  relief="flat", font=("Courier", 11), width=10)
        self.ent_name.grid(row=1, column=0, padx=(0,4), ipady=5)

        tk.Label(row, text="ID", bg=SURFACE, fg=MUTED,
                 font=("Courier", 8)).grid(row=0, column=1, sticky="w")
        self.ent_id = tk.Entry(row, bg=CARD, fg=TEXT, insertbackground=TEXT,
                                relief="flat", font=("Courier", 11), width=5)
        self.ent_id.grid(row=1, column=1, ipady=5)

        tk.Button(sec, text="+ ADD", command=self._add_class,
                  bg=GREEN, fg="#0a0d15", relief="flat",
                  font=("Courier", 9, "bold"), cursor="hand2", pady=7
                  ).pack(fill="x", padx=10, pady=(0,10))

        # Class list
        tk.Label(p, text="CLASSES", bg=BG, fg=MUTED,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=4, pady=(4,2))

        list_frame = tk.Frame(p, bg=SURFACE)
        list_frame.pack(fill="both", expand=True)

        cols = ("ID","Name","Samples")
        self.tree = ttk.Treeview(list_frame, columns=cols, show="headings",
                                   height=14, style="AIT.Treeview")
        self.tree.heading("ID",      text="ID",      anchor="center")
        self.tree.heading("Name",    text="Name",    anchor="w")
        self.tree.heading("Samples", text="Smpl",    anchor="center")
        self.tree.column("ID",      width=38, anchor="center")
        self.tree.column("Name",    width=130, anchor="w")
        self.tree.column("Samples", width=48, anchor="center")

        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        self._style_tree()
        self._refresh_tree()

    def _style_tree(self):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("AIT.Treeview",
            background=SURFACE, foreground=TEXT, fieldbackground=SURFACE,
            rowheight=26, font=("Courier", 10),
            borderwidth=0, relief="flat")
        style.configure("AIT.Treeview.Heading",
            background=CARD, foreground=MUTED,
            font=("Courier", 8, "bold"), borderwidth=0)
        style.map("AIT.Treeview",
            background=[("selected", ACCENT)],
            foreground=[("selected", "white")])

    # ── Train panel ───────────────────────────────────────────

    def _build_train_panel(self):
        p = self.train_panel

        # Selected class
        sel_frame = tk.Frame(p, bg=CARD, pady=12)
        sel_frame.pack(fill="x", pady=(0,8))
        tk.Label(sel_frame, text="SELECTED", bg=CARD, fg=MUTED,
                 font=("Courier", 8, "bold")).pack(anchor="w", padx=14)
        self.lbl_sel = tk.Label(sel_frame, text="— click a class row →",
                                 bg=CARD, fg=MUTED,
                                 font=("Courier", 13, "bold"))
        self.lbl_sel.pack(anchor="w", padx=14)

        # Record section
        rec_frame = tk.Frame(p, bg=CARD)
        rec_frame.pack(fill="x", pady=(0,8))

        top_r = tk.Frame(rec_frame, bg=CARD)
        top_r.pack(fill="x", padx=14, pady=(10,4))

        tk.Label(top_r, text="RECORD", bg=CARD, fg=MUTED,
                 font=("Courier", 8, "bold")).pack(side="left")
        self.lbl_sample_count = tk.Label(top_r, text="", bg=CARD, fg=ACCENT,
                                          font=("Courier", 9))
        self.lbl_sample_count.pack(side="right")

        self.lbl_rec = tk.Label(rec_frame, text="Hold  R  to record  1 second",
                                 bg=CARD, fg=MUTED, font=("Courier", 10))
        self.lbl_rec.pack(padx=14, pady=(0,8))

        # Quick-ID row
        qrow = tk.Frame(rec_frame, bg=CARD)
        qrow.pack(padx=14, pady=(0,8))
        tk.Label(qrow, text="Quick ID:", bg=CARD, fg=MUTED,
                 font=("Courier", 9)).pack(side="left")
        self.ent_qid = tk.Entry(qrow, bg=SURFACE, fg=TEXT, insertbackground=TEXT,
                                 relief="flat", font=("Courier", 11), width=5)
        self.ent_qid.pack(side="left", padx=6, ipady=4)
        tk.Label(qrow, text="← type ID, then hold R", bg=CARD, fg=MUTED,
                 font=("Courier", 8)).pack(side="left")

        self.prog_rec = ttk.Progressbar(rec_frame, length=500, mode="determinate",
                                         style="AIT.Horizontal.TProgressbar")
        self.prog_rec.pack(padx=14, pady=(0,12), fill="x")
        self._style_progressbar()

        # Train section
        trn_frame = tk.Frame(p, bg=CARD)
        trn_frame.pack(fill="x", pady=(0,8))

        top_t = tk.Frame(trn_frame, bg=CARD)
        top_t.pack(fill="x", padx=14, pady=(10,4))
        tk.Label(top_t, text="TRAIN", bg=CARD, fg=MUTED,
                 font=("Courier", 8, "bold")).pack(side="left")

        self.lbl_status = tk.Label(trn_frame, text="No model trained yet.",
                                    bg=CARD, fg=MUTED, font=("Courier", 10))
        self.lbl_status.pack(padx=14, pady=(0,6))

        self.prog_train = ttk.Progressbar(trn_frame, length=500, mode="indeterminate",
                                           style="AIT.Horizontal.TProgressbar")
        self.prog_train.pack(padx=14, pady=(0,8), fill="x")

        self.btn_train = tk.Button(trn_frame, text="▶  TRAIN  MODEL",
                                    command=self._train_start,
                                    bg=ACCENT, fg="white", relief="flat",
                                    font=("Courier", 11, "bold"),
                                    pady=10, cursor="hand2")
        self.btn_train.pack(padx=14, pady=(0,14), fill="x")

    def _style_progressbar(self):
        style = ttk.Style()
        style.configure("AIT.Horizontal.TProgressbar",
            troughcolor=SURFACE, background=ACCENT,
            borderwidth=0, thickness=5)

    # ── Test panel ────────────────────────────────────────────

    def _build_test_panel(self):
        p = self.test_panel

        # Big prediction display
        pred_frame = tk.Frame(p, bg=CARD, pady=16)
        pred_frame.pack(fill="x", pady=(0,8))

        self.lbl_pred_word = tk.Label(pred_frame, text="—",
                                       bg=CARD, fg=GREEN,
                                       font=("Courier", 42, "bold"))
        self.lbl_pred_word.pack()

        self.lbl_pred_conf = tk.Label(pred_frame,
                                       text="Hold  R  to test  ·  Model needed",
                                       bg=CARD, fg=MUTED, font=("Courier", 10))
        self.lbl_pred_conf.pack()

        self.prog_conf = ttk.Progressbar(pred_frame, length=500, mode="determinate",
                                          style="AIT.Horizontal.TProgressbar")
        self.prog_conf.pack(padx=20, pady=(8,0), fill="x")

        # Chart frame
        chart_frame = tk.Frame(p, bg=CARD, bd=0)
        chart_frame.pack(fill="both", expand=True, pady=(0,8))

        self.fig = Figure(figsize=(5, 3.5), dpi=96, facecolor=CARD)
        self.fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01)
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self._draw_empty_chart()

        # Record bar for test mode
        trec = tk.Frame(p, bg=CARD)
        trec.pack(fill="x")
        self.lbl_test_rec = tk.Label(trec, text="Hold  R  to record and predict",
                                      bg=CARD, fg=MUTED, font=("Courier", 10))
        self.lbl_test_rec.pack(pady=(8,4))
        self.prog_test_rec = ttk.Progressbar(trec, length=500, mode="determinate",
                                              style="AIT.Horizontal.TProgressbar")
        self.prog_test_rec.pack(padx=14, pady=(0,10), fill="x")

    def _draw_empty_chart(self):
        self.ax.clear()
        self.ax.set_facecolor(CARD)
        self.ax.text(0.5, 0.5, "Train a model first, then test",
                     transform=self.ax.transAxes,
                     ha="center", va="center",
                     color=MUTED, fontsize=10, fontfamily="monospace")
        self.ax.set_xticks([]); self.ax.set_yticks([])
        for spine in self.ax.spines.values(): spine.set_visible(False)
        try: self.canvas.draw()
        except: pass

    # ── Mode toggle ───────────────────────────────────────────

    def _set_mode(self, mode: str):
        self.mode = mode
        if mode == "train":
            self.train_panel.pack(fill="both", expand=True)
            self.test_panel.pack_forget()
            self.btn_train_mode.config(bg=ACCENT, fg="white")
            self.btn_test_mode.config(bg=CARD, fg=MUTED)
        else:
            self.test_panel.pack(fill="both", expand=True)
            self.train_panel.pack_forget()
            self.btn_test_mode.config(bg=ACCENT, fg="white")
            self.btn_train_mode.config(bg=CARD, fg=MUTED)
            if self.pipeline is None:
                self.lbl_pred_conf.config(
                    text="No model yet  ·  Switch to TRAIN tab first")
        self.root.focus_set()

    # ── Key bindings ──────────────────────────────────────────

    def _bind_keys(self):
        self.root.bind("<KeyPress-r>",   self._on_r_press)
        self.root.bind("<KeyRelease-r>", self._on_r_release)
        self.root.focus_set()

    # ── Class management ──────────────────────────────────────

    def _add_class(self):
        name = self.ent_name.get().strip()
        raw  = self.ent_id.get().strip()
        if not name:
            messagebox.showerror("Error", "Enter a class name."); return
        try:   cid = int(raw)
        except: messagebox.showerror("Error", "ID must be a whole number."); return
        if cid == BGN_CID:
            messagebox.showerror("Error",
                f"ID {BGN_CID} is reserved for BGN (background noise)."); return
        if cid in self.classes:
            messagebox.showerror("Error",
                f"ID {cid} already exists as '{self.classes[cid]}'."); return
        self.classes[cid] = name
        self.recordings.setdefault(cid, [])
        (DATA_DIR / str(cid)).mkdir(parents=True, exist_ok=True)
        self._save_meta()
        self._refresh_tree()
        self.ent_name.delete(0, tk.END)
        self.ent_id.delete(0, tk.END)
        messagebox.showinfo("Added",
            f"Class [{cid}] '{name}' added.\n\nClick the row, then hold  R  to record.")

    def _refresh_tree(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        for cid in sorted(self.classes):
            n = len(self.recordings.get(cid, []))
            self.tree.insert("", "end", values=(cid, self.classes[cid], n))

    def _on_select(self, _event):
        sel = self.tree.selection()
        if not sel: return
        self.current_id = int(self.tree.item(sel[0])["values"][0])
        self._update_sel_label()

    def _update_sel_label(self):
        if self.current_id is None:
            self.lbl_sel.config(text="— click a class row →", fg=MUTED)
            return
        name = self.classes.get(self.current_id, "?")
        n    = len(self.recordings.get(self.current_id, []))
        self.lbl_sel.config(text=f"[{self.current_id}]  {name}", fg=GREEN)
        self.lbl_sample_count.config(text=f"{n} samples")

    def _resolve_id(self) -> bool:
        raw = self.ent_qid.get().strip() if hasattr(self, "ent_qid") else ""
        if raw:
            try:
                cid = int(raw)
                if cid in self.classes:
                    self.current_id = cid
                    self._update_sel_label()
                    return True
            except: pass
        return self.current_id is not None

    # ── Recording ─────────────────────────────────────────────

    def _on_r_press(self, _event):
        if self.r_held or self.recording: return
        if not _HAS_AUDIO:
            messagebox.showerror("No Audio", "sounddevice not installed."); return
        if self.mode == "train":
            if not self._resolve_id():
                messagebox.showwarning("No Class", "Click a class row first."); return
            self.r_held    = True
            self.recording = True
            self.lbl_rec.config(text="● RECORDING …", fg=RED)
            threading.Thread(target=self._rec_train_worker, daemon=True).start()
        else:  # test mode
            if self.pipeline is None:
                messagebox.showwarning("No Model",
                    "Train a model first in the TRAIN tab."); return
            self.r_held    = True
            self.recording = True
            self.lbl_test_rec.config(text="● RECORDING …", fg=RED)
            threading.Thread(target=self._rec_test_worker, daemon=True).start()

    def _on_r_release(self, _event):
        self.r_held = False

    def _rec_train_worker(self):
        cid   = self.current_id
        steps = 50
        try:
            audio = sd.rec(int(SR*DURATION), samplerate=SR,
                           channels=1, dtype="float32")
            for i in range(steps+1):
                self.q.put(("rec_prog", str(i/steps*100)))
                time.sleep(DURATION/steps)
            sd.wait()
            flat = audio.flatten().astype(np.float32)
            rec_dir = DATA_DIR / str(cid)
            rec_dir.mkdir(parents=True, exist_ok=True)
            idx = len(list(rec_dir.glob("*.npy")))
            np.save(rec_dir / f"{idx:05d}.npy", flat)
            self.recordings[cid].append(flat)
        except Exception as e:
            print(f"[AIT] Record error: {e}")
        finally:
            self.q.put(("rec_done", ""))

    def _rec_test_worker(self):
        steps = 50
        try:
            audio = sd.rec(int(SR*DURATION), samplerate=SR,
                           channels=1, dtype="float32")
            for i in range(steps+1):
                # Use prog_test_rec for test mode
                self.prog_test_rec.config(value=i/steps*100)
                time.sleep(DURATION/steps)
            sd.wait()
            flat  = audio.flatten().astype(np.float32)
            feat  = extract_features(flat).reshape(1, -1)
            probs = self.pipeline.predict_proba(feat)[0].tolist()
            self.q.put(("infer", (probs, list(self.cid_list))))
        except Exception as e:
            print(f"[AIT] Test error: {e}")
        finally:
            self.recording = False
            self.r_held    = False
            self.root.after(0, lambda: (
                self.prog_test_rec.config(value=0),
                self.lbl_test_rec.config(
                    text="Hold  R  to record and predict", fg=MUTED)
            ))

    def _rec_done_ui(self):
        self.recording = False
        self.prog_rec.config(value=0)
        self.lbl_rec.config(text="Hold  R  to record  1 second", fg=MUTED)
        self._update_sel_label()
        self._refresh_tree()

    # ── Training ──────────────────────────────────────────────

    def _train_start(self):
        if self.training:
            messagebox.showinfo("Busy", "Training already running."); return
        user_cids = [c for c in self.classes if self.recordings.get(c) and c != BGN_CID]
        if not user_cids:
            messagebox.showerror("No Data",
                "Record at least 1 word first.\n"
                "BGN (background noise) is added automatically."); return
        self.training = True
        self.btn_train.config(state="disabled")
        self.lbl_status.config(text="Starting …", fg=AMBER)
        self.prog_train.start()
        threading.Thread(target=self._train_worker, daemon=True).start()

    def _train_worker(self):
        try:
            self._do_train()
        except Exception:
            tb  = traceback.format_exc()
            print(f"[AIT] Training error:\n{tb}")
            msg = tb.strip().splitlines()[-1]
            self.q.put(("train_err", msg))

    def _do_train(self):
        # Build class list — BGN always at index 0
        user_cids = sorted(c for c in self.classes
                           if self.recordings.get(c) and c != BGN_CID)
        cid_list  = [BGN_CID] + user_cids
        class_map = {BGN_CID: BGN_NAME}
        class_map.update({c: self.classes[c] for c in user_cids})
        label_map = {c: i for i, c in enumerate(cid_list)}
        n_classes = len(cid_list)

        self.q.put(("train_msg", "Extracting MFCC features …"))

        X_list: list[np.ndarray] = []
        y_list: list[int]        = []

        # User recordings
        for cid in user_cids:
            for audio in self.recordings[cid]:
                X_list.append(extract_features(audio))
                y_list.append(label_map[cid])

        # BGN synthetic samples — always added
        n_bgn = max(20, len(X_list) // max(len(user_cids), 1))
        rng   = np.random.RandomState(42)
        t_arr = np.linspace(0, 1, SR, dtype=np.float32)
        print(f"[AIT] Adding {n_bgn} BGN samples")
        for i in range(n_bgn):
            amp   = rng.uniform(0.001, 0.025)
            noise = rng.randn(SR).astype(np.float32) * amp
            if i % 3 == 0:
                freq  = rng.uniform(20, 150)
                noise += amp * 0.4 * np.sin(2 * np.pi * freq * t_arr)
            X_list.append(extract_features(noise))
            y_list.append(label_map[BGN_CID])

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int32)
        n_samples = len(X)

        self.q.put(("train_msg",
            f"Training … {n_samples} samples, {n_classes} classes"))

        pipeline = Pipeline([
            ("sc",  StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=HIDDEN,
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=False,
                learning_rate_init=0.001,
                alpha=1e-4,
                random_state=42,
                verbose=False,
            )),
        ])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            pipeline.fit(X, y)

        # CV if enough samples
        cv_msg        = ""
        min_per_class = int(min((y == i).sum() for i in range(n_classes)))
        if min_per_class >= 3:
            nf = min(5, min_per_class)
            self.q.put(("train_msg", f"Cross-validating ({nf} folds) …"))
            cv = StratifiedKFold(n_splits=nf, shuffle=True, random_state=42)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                scores = cross_val_score(pipeline, X, y, cv=cv,
                                         scoring="accuracy", n_jobs=1)
            cv_msg = f"  CV={scores.mean():.3f}±{scores.std():.3f}"

        train_acc   = float((pipeline.predict(X) == y).mean())
        label_names = [class_map[cid_list[i]] for i in range(n_classes)]
        print("[AIT] Classification report:")
        print(classification_report(
            y, pipeline.predict(X), target_names=label_names, zero_division=0))

        # Save
        joblib.dump(pipeline, MODEL_PKL)
        meta = {
            "cid_list":     cid_list,
            "classes":      {str(c): class_map[c] for c in cid_list},
            "n_classes":    n_classes,
            "input_dim":    INPUT_DIM,
            "hidden_sizes": list(HIDDEN),
            "scaler":       {"mean": pipeline.named_steps["sc"].mean_.tolist(),
                             "scale": pipeline.named_steps["sc"].scale_.tolist()},
            "layers":       [{"W": clf.tolist(), "b": bi.tolist()}
                             for clf, bi in zip(
                                 pipeline.named_steps["clf"].coefs_,
                                 pipeline.named_steps["clf"].intercepts_)],
        }
        WEIGHTS_JSON.write_text(json.dumps(meta, indent=2))
        print(f"[AIT] Saved {MODEL_PKL}")

        # Update app state
        self.pipeline = pipeline
        self.cid_list = cid_list

        msg = (f"✓ {n_classes} classes  |  {n_samples} samples  |  "
               f"acc={train_acc:.3f}{cv_msg}")
        self.q.put(("train_done", msg))

    def _train_done_ui(self, msg: str):
        self.training = False
        self.prog_train.stop()
        self.btn_train.config(state="normal")
        self.lbl_status.config(text=msg, fg=GREEN)
        messagebox.showinfo("Training Complete",
            msg + "\n\nSwitch to TEST tab and hold R to test your model.")

    def _train_err_ui(self, msg: str):
        self.training = False
        self.prog_train.stop()
        self.btn_train.config(state="normal")
        self.lbl_status.config(text=f"Error: {msg}", fg=RED)
        messagebox.showerror("Training Failed", msg)

    # ── Prediction display ────────────────────────────────────

    def _show_prediction(self, probs: list, cid_list: list):
        if not cid_list or len(cid_list) != len(probs):
            return

        best_i    = int(np.argmax(probs))
        best_cid  = cid_list[best_i] if best_i < len(cid_list) else -1
        best_name = self.classes.get(best_cid, BGN_NAME if best_cid == BGN_CID else "?")
        conf      = probs[best_i] * 100.0

        # If BGN wins — show it but colour differently
        if best_cid == BGN_CID:
            self.lbl_pred_word.config(text="BGN", fg=AMBER)
            self.lbl_pred_conf.config(
                text=f"Background noise  ·  {conf:.1f}% confidence", fg=AMBER)
        else:
            self.lbl_pred_word.config(text=best_name, fg=GREEN)
            self.lbl_pred_conf.config(
                text=f"Confidence: {conf:.1f}%  ·  Class ID: {best_cid}", fg=TEXT)

        self.prog_conf.config(value=conf)
        self._draw_chart(probs, cid_list)

    def _draw_chart(self, probs: list, cid_list: list):
        self.ax.clear()
        self.ax.set_facecolor(CARD)
        self.fig.set_facecolor(CARD)

        n      = len(cid_list)
        labels = [self.classes.get(c, BGN_NAME if c == BGN_CID else str(c))
                  for c in cid_list]
        colors = [PAL[i % len(PAL)] for i in range(n)]

        # Horizontal bar chart — more readable with many classes
        y_pos = list(range(n))
        bars  = self.ax.barh(y_pos, probs, color=colors,
                              height=0.65, left=0)

        best_i = int(np.argmax(probs))
        self.ax.barh(best_i, probs[best_i],
                     color=colors[best_i], height=0.65,
                     edgecolor="white", linewidth=1.5)

        self.ax.set_yticks(y_pos)
        self.ax.set_yticklabels(labels, fontfamily="monospace",
                                 fontsize=9, color=TEXT)
        self.ax.set_xlim(0, 1)
        self.ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        self.ax.set_xticklabels(["0%","25%","50%","75%","100%"],
                                  fontsize=7, color=MUTED)
        self.ax.tick_params(colors=MUTED, length=0)

        for spine in self.ax.spines.values():
            spine.set_color(BORDER)

        # Value labels on bars
        for i, (bar, p) in enumerate(zip(bars, probs)):
            if p > 0.04:
                self.ax.text(p + 0.01, i, f"{p:.0%}",
                             va="center", fontsize=7,
                             color=TEXT, fontfamily="monospace")

        self.ax.set_title(
            labels[best_i] + f"  ({probs[best_i]:.0%})",
            color=TEXT, fontfamily="monospace", fontsize=11, pad=6)

        self.fig.tight_layout(pad=0.8)
        try: self.canvas.draw()
        except: pass


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not _HAS_AUDIO:
        print("[AIT] WARNING: sounddevice not installed.")
        print("      pip install sounddevice")

    root = tk.Tk()
    app  = AIT(root)
    root.mainloop()
