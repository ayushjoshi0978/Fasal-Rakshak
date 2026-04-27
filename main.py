import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.history_manager import save_to_history, load_history

# ─── Colors & Fonts ───────────────────────────────────────────
BG_DARK     = "#1a2e1a"
BG_CARD     = "#243324"
BG_INPUT    = "#1e2b1e"
GREEN_MAIN  = "#4caf50"
GREEN_LIGHT = "#81c784"
GREEN_PALE  = "#c8e6c9"
YELLOW      = "#ffb74d"
RED_ALERT   = "#ef5350"
WHITE       = "#f0f4f0"
MUTED       = "#8fa98f"

FN          = ("Segoe UI", 11)
FN_B        = ("Segoe UI", 11, "bold")
FN_L        = ("Segoe UI", 15, "bold")
FN_S        = ("Segoe UI", 9)
FH          = ("Nirmala UI", 11)
FH_B        = ("Nirmala UI", 13, "bold")
FH_L        = ("Nirmala UI", 15, "bold")


class FasalRakshak(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("फसल रक्षक — Crop Disease Detector (Offline AI)")
        self.geometry("940x700")
        self.minsize(860, 620)
        self.configure(bg=BG_DARK)

        self.current_image_path = None
        self.photo_ref = None
        self.analyzing = False
        self._last_result = None

        self._build_ui()
        self._check_model_on_startup()

    # ─── Startup model check ──────────────────────────────────
    def _check_model_on_startup(self):
        from utils.ml_classifier import is_model_trained
        if not is_model_trained():
            self.status_dot.config(text="● Training model...", fg=YELLOW)
            threading.Thread(target=self._train_bg, daemon=True).start()
        else:
            self.status_dot.config(text="● Model ready (Offline)", fg=GREEN_MAIN)

    def _train_bg(self):
        try:
            from utils.ml_classifier import train_model
            info = train_model()
            self.after(0, lambda: self.status_dot.config(
                text=f"● Model ready — {info['classes']} diseases ({info['samples']} samples)",
                fg=GREEN_MAIN))
        except Exception as e:
            self.after(0, lambda: self.status_dot.config(
                text=f"● Model error: {str(e)[:40]}", fg=RED_ALERT))

    # ─── Build UI ─────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg="#152515", pady=12)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🌾", font=("Segoe UI Emoji", 22),
                 bg="#152515", fg=GREEN_LIGHT).pack(side="left", padx=(18,6))
        tframe = tk.Frame(hdr, bg="#152515")
        tframe.pack(side="left")
        tk.Label(tframe, text="फसल रक्षक", font=FH_B,
                 bg="#152515", fg=GREEN_LIGHT).pack(anchor="w")
        tk.Label(tframe, text="Fasal Rakshak — Offline AI  |  No Internet Required",
                 font=FN_S, bg="#152515", fg=MUTED).pack(anchor="w")

        self.status_dot = tk.Label(hdr, text="● Starting...",
                                    font=FN_S, bg="#152515", fg=YELLOW)
        self.status_dot.pack(side="right", padx=20)

        # Notebook
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("T.TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("T.TNotebook.Tab", background=BG_CARD, foreground=MUTED,
                         padding=[16, 8], font=FN)
        style.map("T.TNotebook.Tab",
                  background=[("selected", BG_DARK)],
                  foreground=[("selected", GREEN_LIGHT)])

        nb = ttk.Notebook(self, style="T.TNotebook")
        nb.pack(fill="both", expand=True, padx=12, pady=(10,12))

        self.tab_detect  = tk.Frame(nb, bg=BG_DARK)
        self.tab_history = tk.Frame(nb, bg=BG_DARK)
        self.tab_train   = tk.Frame(nb, bg=BG_DARK)
        self.tab_about   = tk.Frame(nb, bg=BG_DARK)

        nb.add(self.tab_detect,  text="  📷 Detect Disease  ")
        nb.add(self.tab_history, text="  📋 History  ")
        nb.add(self.tab_train,   text="  🧠 Train Model  ")
        nb.add(self.tab_about,   text="  ℹ️ About  ")

        self._build_detect_tab()
        self._build_train_tab()
        self._build_about_tab()
        self._refresh_history_tab()

    # ─── Detect Tab ───────────────────────────────────────────
    def _build_detect_tab(self):
        main = tk.Frame(self.tab_detect, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=8, pady=8)

        # Left panel
        left = tk.Frame(main, bg=BG_CARD, width=340)
        left.pack(side="left", fill="y", padx=(0,8))
        left.pack_propagate(False)

        tk.Label(left, text="फोटो अपलोड करें", font=FH_B,
                 bg=BG_CARD, fg=WHITE).pack(pady=(16,2))
        tk.Label(left, text="Upload Crop Leaf Photo",
                 font=FN_S, bg=BG_CARD, fg=MUTED).pack()

        # Image preview
        self.img_frame = tk.Frame(left, bg=BG_INPUT, width=290, height=270)
        self.img_frame.pack(padx=20, pady=10)
        self.img_frame.pack_propagate(False)
        self.img_label = tk.Label(self.img_frame, bg=BG_INPUT,
                                   text="📷\n\nकोई फोटो नहीं\nNo image selected",
                                   font=FH, fg=MUTED, justify="center")
        self.img_label.pack(expand=True)

        self._btn(left, "📁  फोटो चुनें / Browse", self._browse,
                  GREEN_MAIN).pack(fill="x", padx=20, pady=(4,3))

        # Crop selector
        tk.Label(left, text="फसल का प्रकार (Crop Type):",
                 font=FH, bg=BG_CARD, fg=GREEN_PALE).pack(padx=20, anchor="w", pady=(8,2))
        self.crop_var = tk.StringVar(value="Auto Detect")
        crops = ["Auto Detect", "Tomato (टमाटर)", "Potato (आलू)",
                 "Wheat (गेहूं)", "Rice (धान)", "Corn/Maize (मक्का)"]
        ttk.Combobox(left, textvariable=self.crop_var, values=crops,
                     state="readonly", font=FN, width=30).pack(padx=20, fill="x")

        # Language
        lf = tk.Frame(left, bg=BG_CARD)
        lf.pack(padx=20, fill="x", pady=(8,0))
        tk.Label(lf, text="भाषा:", font=FN_S, bg=BG_CARD, fg=MUTED).pack(side="left")
        self.lang_var = tk.StringVar(value="Hindi")
        for lang in ["Hindi", "English", "Hinglish"]:
            tk.Radiobutton(lf, text=lang, variable=self.lang_var,
                           value=lang, bg=BG_CARD, fg=WHITE,
                           selectcolor=BG_INPUT, activebackground=BG_CARD,
                           font=FN_S).pack(side="left", padx=6)

        self.analyze_btn = self._btn(left, "🔍  रोग पहचानें / Analyze",
                                      self._start_analysis, GREEN_MAIN,
                                      font=("Segoe UI", 12, "bold"), pady=10)
        self.analyze_btn.pack(fill="x", padx=20, pady=12)

        # Right panel — results
        right = tk.Frame(main, bg=BG_DARK)
        right.pack(side="left", fill="both", expand=True)

        res_hdr = tk.Frame(right, bg=BG_CARD)
        res_hdr.pack(fill="x", pady=(0,6))
        tk.Label(res_hdr, text="विश्लेषण परिणाम / Analysis Result",
                 font=FH_B, bg=BG_CARD, fg=WHITE, pady=10, padx=16).pack(side="left")
        self.conf_label = tk.Label(res_hdr, text="", font=FN_S,
                                    bg=BG_CARD, fg=GREEN_MAIN)
        self.conf_label.pack(side="right", padx=16)

        # Disease name
        self.disease_frame = tk.Frame(right, bg=BG_CARD)
        self.disease_frame.pack(fill="x", pady=(0,6))
        self.disease_label = tk.Label(self.disease_frame,
                                       text="— अभी तक कोई विश्लेषण नहीं —",
                                       font=FH_L, bg=BG_CARD, fg=MUTED,
                                       wraplength=520, justify="center", pady=14)
        self.disease_label.pack(fill="x")

        # Severity bar
        self.sev_frame = tk.Frame(right, bg=BG_DARK)
        self.sev_frame.pack(fill="x", pady=(0,6))

        # Top 3 predictions frame
        self.top3_frame = tk.Frame(right, bg=BG_CARD)
        self.top3_frame.pack(fill="x", pady=(0,6))

        # Result text
        outer = tk.Frame(right, bg=BG_CARD)
        outer.pack(fill="both", expand=True)
        sb = tk.Scrollbar(outer)
        sb.pack(side="right", fill="y")
        self.result_text = tk.Text(outer, bg=BG_INPUT, fg=WHITE, font=FH,
                                    wrap="word", relief="flat", bd=0,
                                    yscrollcommand=sb.set, padx=14, pady=14)
        self.result_text.pack(fill="both", expand=True)
        sb.config(command=self.result_text.yview)
        self._set_result(
            "यहाँ आपकी फसल का विश्लेषण दिखेगा।\n\n"
            "Steps:\n"
            "1. फसल की पत्ती की फोटो चुनें\n"
            "2. फसल का प्रकार चुनें\n"
            "3. 'रोग पहचानें' दबाएं\n\n"
            "✅ यह पूरी तरह Offline काम करता है!\n"
            "   No Internet | No API Key | No Credits"
        )

        self._btn(right, "💾  इतिहास में सहेजें / Save",
                  self._save_result, "#37474f").pack(fill="x", pady=(6,0))

    # ─── Train Tab ────────────────────────────────────────────
    def _build_train_tab(self):
        f = tk.Frame(self.tab_train, bg=BG_DARK)
        f.pack(expand=True, fill="both", padx=30, pady=20)

        tk.Label(f, text="🧠 Model Training", font=FN_L,
                 bg=BG_DARK, fg=GREEN_LIGHT).pack(pady=(20,4))
        tk.Label(f, text="Train the AI model with your own images or use built-in data",
                 font=FN_S, bg=BG_DARK, fg=MUTED).pack(pady=(0,20))

        # Status card
        info = tk.Frame(f, bg=BG_CARD)
        info.pack(fill="x", pady=(0,16))
        tk.Label(info, text="📊 Current Model Status",
                 font=FN_B, bg=BG_CARD, fg=WHITE, pady=10, padx=16).pack(anchor="w")
        self.train_status = tk.Label(info,
            text="Checking...", font=FH, bg=BG_CARD, fg=MUTED,
            padx=16, pady=8, justify="left")
        self.train_status.pack(anchor="w")
        self.after(1000, self._update_train_status)

        # Option 1 — retrain with built-in data
        card1 = tk.Frame(f, bg=BG_CARD)
        card1.pack(fill="x", pady=(0,10))
        tk.Label(card1, text="Option 1 — Quick Train (Built-in Synthetic Data)",
                 font=FN_B, bg=BG_CARD, fg=GREEN_LIGHT, padx=16, pady=8).pack(anchor="w")
        tk.Label(card1,
                 text="Uses AI-generated feature profiles for 14 disease classes.\n"
                      "Good accuracy, trains in seconds. No dataset needed.",
                 font=FN_S, bg=BG_CARD, fg=MUTED, padx=16, justify="left").pack(anchor="w")
        self._btn(card1, "⚡  Quick Train Now", self._quick_train,
                  GREEN_MAIN).pack(padx=16, pady=10, anchor="w")

        # Option 2 — train with real images
        card2 = tk.Frame(f, bg=BG_CARD)
        card2.pack(fill="x", pady=(0,10))
        tk.Label(card2, text="Option 2 — Train with Real Images (Better Accuracy)",
                 font=FN_B, bg=BG_CARD, fg=GREEN_LIGHT, padx=16, pady=8).pack(anchor="w")
        tk.Label(card2,
                 text="Select a folder with subfolders named by disease class.\n"
                      "Download PlantVillage dataset from Kaggle for best results.\n"
                      "Folder structure: dataset/Tomato___Early_blight/*.jpg",
                 font=FN_S, bg=BG_CARD, fg=MUTED, padx=16, justify="left").pack(anchor="w")

        btn_row = tk.Frame(card2, bg=BG_CARD)
        btn_row.pack(padx=16, pady=10, anchor="w")
        self._btn(btn_row, "📁  Select Dataset Folder",
                  self._select_dataset, "#378ADD").pack(side="left", padx=(0,10))
        self.dataset_path_label = tk.Label(btn_row, text="No folder selected",
                                            font=FN_S, bg=BG_CARD, fg=MUTED)
        self.dataset_path_label.pack(side="left")

        self._btn(card2, "🚀  Train with Real Images",
                  self._real_train, "#185FA5").pack(padx=16, pady=(0,12), anchor="w")

        # Progress bar
        self.train_progress = ttk.Progressbar(f, mode="indeterminate", length=400)
        self.train_progress.pack(pady=(10,4))
        self.train_log = tk.Label(f, text="", font=FN_S, bg=BG_DARK, fg=GREEN_LIGHT)
        self.train_log.pack()

        self._dataset_folder = None

    def _update_train_status(self):
        from utils.ml_classifier import is_model_trained, MODEL_PATH
        import os
        if is_model_trained():
            size = os.path.getsize(MODEL_PATH) // 1024
            self.train_status.config(
                text=f"✅ Model trained and ready\n"
                     f"   File: model/classifier.pkl ({size} KB)\n"
                     f"   Diseases: 14 classes (Tomato, Potato, Wheat, Rice, Corn)",
                fg=GREEN_MAIN)
        else:
            self.train_status.config(
                text="⚠️ Model not trained yet. Click 'Quick Train Now'.",
                fg=YELLOW)

    def _select_dataset(self):
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        if folder:
            self._dataset_folder = folder
            short = "..." + folder[-35:] if len(folder) > 38 else folder
            self.dataset_path_label.config(text=short, fg=GREEN_LIGHT)

    def _quick_train(self):
        self.train_progress.start(10)
        self.train_log.config(text="Training model with synthetic data...")
        threading.Thread(target=self._do_train, args=(None,), daemon=True).start()

    def _real_train(self):
        if not self._dataset_folder:
            messagebox.showwarning("⚠️", "Please select a dataset folder first.")
            return
        self.train_progress.start(10)
        self.train_log.config(text=f"Training with images from: {self._dataset_folder[:40]}...")
        threading.Thread(target=self._do_train,
                         args=(self._dataset_folder,), daemon=True).start()

    def _do_train(self, dataset_dir):
        try:
            from utils.ml_classifier import train_model
            info = train_model(dataset_dir)
            self.after(0, self._train_done, info)
        except Exception as e:
            self.after(0, self._train_error, str(e))

    def _train_done(self, info):
        self.train_progress.stop()
        self.train_log.config(
            text=f"✅ Done! {info['classes']} classes, {info['samples']} samples "
                 f"({info['real_images']} real images)",
            fg=GREEN_LIGHT)
        self.status_dot.config(text="● Model ready (Offline)", fg=GREEN_MAIN)
        self._update_train_status()

    def _train_error(self, err):
        self.train_progress.stop()
        self.train_log.config(text=f"❌ Error: {err[:60]}", fg=RED_ALERT)

    # ─── About Tab ────────────────────────────────────────────
    def _build_about_tab(self):
        f = tk.Frame(self.tab_about, bg=BG_DARK)
        f.pack(expand=True)
        tk.Label(f, text="🌾", font=("Segoe UI Emoji", 44), bg=BG_DARK).pack(pady=(40,4))
        tk.Label(f, text="फसल रक्षक v2.0", font=FH_L, bg=BG_DARK, fg=GREEN_LIGHT).pack()
        tk.Label(f, text="Fully Offline AI — No API — No Internet",
                 font=FN_S, bg=BG_DARK, fg=MUTED).pack(pady=(2,20))
        txt = (
            "यह ऐप बिना internet के काम करता है।\n"
            "Machine Learning से फसल के रोग पहचानता है।\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "🤖  ML Model: KNN Classifier (scikit-learn)\n"
            "🎨  Features: Color Histogram + Texture Analysis\n"
            "🌱  Diseases: 14 classes across 5 crops\n"
            "🐍  Built with: Python + Tkinter + PIL + sklearn\n"
            "💾  Data: Local only — nothing sent online\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "📦  Better accuracy: Download PlantVillage dataset\n"
            "     from kaggle.com and use 'Train with Real Images'\n\n"
            "🌾  Supported crops: Tomato, Potato, Wheat, Rice, Corn"
        )
        tk.Label(f, text=txt, font=FH, bg=BG_DARK, fg=WHITE,
                 justify="center", wraplength=520).pack()

    # ─── History Tab ─────────────────────────────────────────
    def _refresh_history_tab(self):
        for w in self.tab_history.winfo_children():
            w.destroy()
        tk.Label(self.tab_history, text="पिछले विश्लेषण / Scan History",
                 font=FH_B, bg=BG_DARK, fg=WHITE, pady=10).pack()
        history = load_history()
        if not history:
            tk.Label(self.tab_history,
                     text="अभी कोई इतिहास नहीं है।\nNo scans saved yet.",
                     font=FH, bg=BG_DARK, fg=MUTED, pady=50).pack()
            return
        canvas = tk.Canvas(self.tab_history, bg=BG_DARK, highlightthickness=0)
        sb = tk.Scrollbar(self.tab_history, orient="vertical", command=canvas.yview)
        frm = tk.Frame(canvas, bg=BG_DARK)
        frm.bind("<Configure>", lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=frm, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        sb.pack(side="right", fill="y")
        for item in reversed(history):
            card = tk.Frame(frm, bg=BG_CARD)
            card.pack(fill="x", padx=8, pady=4)
            color = item.get("color", GREEN_MAIN)
            tk.Label(card, text=f"  {item.get('date','')}",
                     font=FN_S, bg=BG_CARD, fg=MUTED).pack(anchor="w", padx=8, pady=(8,0))
            tk.Label(card, text=item.get("disease_name",""),
                     font=FH_B, bg=BG_CARD, fg=color).pack(anchor="w", padx=8)
            tk.Label(card,
                     text=f"Confidence: {item.get('confidence','')}%  |  "
                          f"Severity: {item.get('severity','')}",
                     font=FN_S, bg=BG_CARD, fg=MUTED).pack(anchor="w", padx=8, pady=(0,8))

    # ─── Analyze Logic ────────────────────────────────────────
    def _browse(self):
        path = filedialog.askopenfilename(
            title="फोटो चुनें",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if path:
            self.current_image_path = path
            img = Image.open(path)
            img.thumbnail((290, 270), Image.LANCZOS)
            self.photo_ref = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.photo_ref, text="")
            self.status_dot.config(text="● Image loaded", fg=GREEN_MAIN)

    def _start_analysis(self):
        if not self.current_image_path:
            messagebox.showwarning("⚠️", "पहले फोटो चुनें!\nSelect an image first.")
            return
        if self.analyzing:
            return
        self.analyzing = True
        self.analyze_btn.config(text="⏳ Analyzing...", state="disabled")
        self.status_dot.config(text="● Analyzing...", fg=YELLOW)
        self.disease_label.config(text="विश्लेषण हो रहा है...", fg=YELLOW)
        self._set_result("🔄 ML model image analyze कर रही है...\nकृपया प्रतीक्षा करें...")
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            from utils.ml_classifier import predict
            result = predict(self.current_image_path)
            self.after(0, self._show_result, result)
        except Exception as e:
            self.after(0, self._show_error, str(e))

    def _show_result(self, result: dict):
        self.analyzing = False
        self.analyze_btn.config(text="🔍  रोग पहचानें / Analyze", state="normal")

        info       = result["disease_info"]
        confidence = result["confidence"]
        top3       = result["top3"]
        lang       = self.lang_var.get()

        color = info.get("color", GREEN_MAIN)
        severity = info.get("severity", "")

        if lang == "Hindi":
            name = info.get("name_hindi", "")
        else:
            name = info.get("name_english", "")

        # Confidence color
        if confidence >= 70:
            conf_color = GREEN_MAIN
            conf_text  = f"Confidence: {confidence}% ✓"
        elif confidence >= 50:
            conf_color = YELLOW
            conf_text  = f"Confidence: {confidence}% (Moderate)"
        else:
            conf_color = RED_ALERT
            conf_text  = f"Confidence: {confidence}% (Low — try clearer photo)"

        self.status_dot.config(text="● Analysis complete", fg=GREEN_MAIN)
        self.disease_label.config(text=f"🦠 {name}", fg=color)
        self.conf_label.config(text=conf_text, fg=conf_color)

        # Severity badge
        for w in self.sev_frame.winfo_children():
            w.destroy()
        sev_colors = {"High": RED_ALERT, "Medium": YELLOW, "Low": GREEN_MAIN, "None": GREEN_MAIN}
        sc = sev_colors.get(severity, MUTED)
        if severity and severity != "None":
            tk.Label(self.sev_frame,
                     text=f"  ⚠️ गंभीरता / Severity: {severity}  ",
                     font=FN_B, bg=sc, fg=BG_DARK, pady=4).pack(padx=8, anchor="w")

        # Top 3 predictions
        for w in self.top3_frame.winfo_children():
            w.destroy()
        tk.Label(self.top3_frame, text="Top predictions:",
                 font=FN_S, bg=BG_CARD, fg=MUTED, padx=10, pady=4).pack(anchor="w")
        row = tk.Frame(self.top3_frame, bg=BG_CARD)
        row.pack(fill="x", padx=10, pady=(0,6))
        for i, (lkey, prob) in enumerate(top3):
            short = lkey.replace("___", " — ").replace("_", " ")
            clr = GREEN_MAIN if i == 0 else MUTED
            tk.Label(row, text=f"#{i+1} {short} ({prob:.1f}%)",
                     font=FN_S, bg=BG_CARD, fg=clr).pack(side="left", padx=(0,16))

        # Build full result text
        if lang == "Hindi":
            text = self._build_hindi_result(info, confidence)
        elif lang == "English":
            text = self._build_english_result(info, confidence)
        else:
            text = self._build_hinglish_result(info, confidence)

        self._set_result(text)

        self._last_result = {
            "disease_name": name,
            "confidence":   confidence,
            "severity":     severity,
            "color":        color,
            "image_path":   self.current_image_path
        }
        self._refresh_history_tab()

    def _build_hindi_result(self, info, confidence):
        sev_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢", "None": "✅"}.get(
            info.get("severity",""), "⚪")
        return (
            f"🌿 फसल: {info.get('name_hindi','')}\n"
            f"{sev_emoji} गंभीरता: {info.get('severity','')}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🔍 लक्षण:\n{info.get('symptoms_hindi','')}\n\n"
            f"💊 रासायनिक उपचार:\n{info.get('treatment_hindi','')}\n\n"
            f"🌿 जैविक/देसी उपाय:\n{info.get('organic_hindi','')}\n\n"
            f"🛡️ बचाव:\n{info.get('prevention_hindi','')}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 इलाज न हो तो नुकसान: {info.get('loss_pct','')}\n\n"
            f"👨‍🌾 किसान को सलाह:\n{info.get('advice_hindi','')}\n\n"
            f"🤖 AI Confidence: {confidence}%"
        )

    def _build_english_result(self, info, confidence):
        return (
            f"🌿 Crop Disease: {info.get('name_english','')}\n"
            f"⚠️ Severity: {info.get('severity','')}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🔍 Symptoms:\n{info.get('symptoms_hindi','').replace('•','•')}\n\n"
            f"💊 Chemical Treatment:\n{info.get('treatment_hindi','')}\n\n"
            f"🌿 Organic Remedy:\n{info.get('organic_hindi','')}\n\n"
            f"🛡️ Prevention:\n{info.get('prevention_hindi','')}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Yield Loss if Untreated: {info.get('loss_pct','')}\n\n"
            f"👨‍🌾 Farmer Advice:\n{info.get('advice_hindi','')}\n\n"
            f"🤖 AI Confidence: {confidence}%"
        )

    def _build_hinglish_result(self, info, confidence):
        return (
            f"🌿 Fasal: {info.get('name_hindi','')}\n"
            f"⚠️ Gambhirta: {info.get('severity','')}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🔍 Lakshan:\n{info.get('symptoms_hindi','')}\n\n"
            f"💊 Ilaaj:\n{info.get('treatment_hindi','')}\n\n"
            f"🌿 Desi Upay:\n{info.get('organic_hindi','')}\n\n"
            f"🛡️ Bachav:\n{info.get('prevention_hindi','')}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Nuksan agar ilaaj na ho: {info.get('loss_pct','')}\n\n"
            f"👨‍🌾 Salah:\n{info.get('advice_hindi','')}\n\n"
            f"🤖 AI Confidence: {confidence}%"
        )

    def _show_error(self, err):
        self.analyzing = False
        self.analyze_btn.config(text="🔍  रोग पहचानें / Analyze", state="normal")
        self.status_dot.config(text="● Error", fg=RED_ALERT)
        self.disease_label.config(text="⚠️ Error", fg=RED_ALERT)
        self._set_result(f"❌ Error:\n\n{err}")

    def _set_result(self, text):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", text)
        self.result_text.config(state="disabled")

    def _save_result(self):
        if not self._last_result:
            messagebox.showinfo("ℹ️", "पहले विश्लेषण करें!\nAnalyze first.")
            return
        save_to_history(self._last_result)
        self._refresh_history_tab()
        messagebox.showinfo("✅", "इतिहास में सहेज लिया!\nSaved to history.")

    def _btn(self, parent, text, cmd, color, font=FN_B, pady=8):
        return tk.Button(parent, text=text, command=cmd,
                         bg=color, fg=WHITE, relief="flat",
                         font=font, pady=pady,
                         activebackground=GREEN_LIGHT,
                         activeforeground=BG_DARK,
                         cursor="hand2", bd=0)


if __name__ == "__main__":
    app = FasalRakshak()
    app.mainloop()
