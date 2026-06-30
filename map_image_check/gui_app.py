"""
Tkinter GUI for terrain-map scanning.

Run from project root:
    python -m map_image_check.gui_app
"""

from __future__ import annotations

import base64
import csv
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np

from .ad_remote import (
    build_unc_roots,
    default_remote_shares,
    remote_share_choices,
)
from .app_info import APP_AUTHOR, APP_DESCRIPTION, APP_NAME, APP_VERSION
from .detector import MAP_SCORE_THRESHOLD, imread_unicode
from .hybrid_pipeline import (
    DEFAULT_T_ACCEPT,
    DEFAULT_T_LOW,
    DEFAULT_T_REJECT,
    HybridConfig,
    classify_image_path,
)
from .image_store import (
    ImageStore,
    ImageTooLargeForStoreError,
    MAX_STORED_IMAGE_BYTES,
    ML_FILTER_ALL,
    ML_FILTER_GT90,
    ML_FILTER_RANGE_80_90,
    StoredImageRecord,
    format_data_size,
    lookup_path_in_index,
    matches_ml_score_filter,
    parse_feature_summary,
    record_ml_score,
)
from .llm_analysis import (
    DEFAULT_LOCAL_MODEL,
    DEFAULT_OLLAMA_BASE,
    analyze_image_with_local_llm,
    classify_map_yes_no,
    format_health_check_message,
    normalize_ollama_base_url,
    ollama_health_check,
)
from .ml_classifier import MapMlClassifier
from .scan_drives import (
    _DEFAULT_OUTPUT,
    _MIN_FILE_BYTES,
    _MIN_IMAGE_HEIGHT,
    _MIN_IMAGE_WIDTH,
    _PROGRESS_EVERY,
    _fixed_drive_roots,
    _passes_size_filters,
    _walk_images,
)
from .search_settings_dialog import SearchSettingsDialog

_PREVIEW_MAX_W = 900
_PREVIEW_MAX_H = 720
_DB_FILTER_ALL = "Все компьютеры"
_DB_FILTER_LOCAL_LABEL = "Локально"
_DB_ML_FILTER_LABELS = {
    ML_FILTER_ALL: "Все ML",
    ML_FILTER_GT90: "ML > 90%",
    ML_FILTER_RANGE_80_90: "ML 80–90%",
}
_THEME_PALETTES = {
    "light": {
        "base_bg": "#eef2f7",
        "card_bg": "#ffffff",
        "accent": "#2563eb",
        "text": "#0f172a",
        "muted": "#475569",
        "border": "#d7dee8",
        "select_bg": "#dbeafe",
        "select_fg": "#0f172a",
        "input_bg": "#ffffff",
    },
    "dark": {
        "base_bg": "#0f172a",
        "card_bg": "#111827",
        "accent": "#60a5fa",
        "text": "#e5eefb",
        "muted": "#94a3b8",
        "border": "#243244",
        "select_bg": "#1d4ed8",
        "select_fg": "#eff6ff",
        "input_bg": "#0b1220",
    },
}


class MapScannerGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1440x920")
        self.minsize(1120, 760)
        self.configure(bg="#eef2f7")

        self._event_queue: queue.Queue[tuple] = queue.Queue()
        self._scan_thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._found_paths: list[str] = []
        self._folder_paths: list[str] = []
        self._ad_computers: list[str] = []
        self._ad_computer_vars: dict[str, tk.BooleanVar] = {}
        self._ad_computer_is_server: dict[str, bool] = {}
        self._preview_image: tk.PhotoImage | None = None
        self._preview_path: Path | None = None
        self._selected_record: StoredImageRecord | None = None
        self._found_image_ids: dict[str, int] = {}
        self._viewing_database = False
        self._db_all_records: list[StoredImageRecord] = []
        self._scan_mode_var = tk.StringVar(value="all")
        self._drive_vars: dict[str, tk.BooleanVar] = {}
        self._remote_share_vars: dict[str, tk.BooleanVar] = {}
        self._theme_name = "light"
        self._theme_var = tk.StringVar(value="light")
        self._image_store = ImageStore()
        self._ml_classifier = MapMlClassifier()
        self._settings_dialog: SearchSettingsDialog | None = None

        self._status_var = tk.StringVar(value="Готово к сканированию.")
        self._count_var = tk.StringVar(value="0")
        self._selected_path_var = tk.StringVar(
            value="Выберите файл из списка, чтобы посмотреть предпросмотр."
        )
        self._ad_status_var = tk.StringVar(value="Список компьютеров AD еще не загружен.")
        self._mode_summary_var = tk.StringVar(value="Источник: все локальные диски")
        self._db_status_var = tk.StringVar(value=self._db_status_text())
        self._llm_status_var = tk.StringVar(value="LLM-анализ еще не выполнялся.")
        self._llm_model_var = tk.StringVar(value=DEFAULT_LOCAL_MODEL)
        self._llm_endpoint_var = tk.StringVar(value=DEFAULT_OLLAMA_BASE)
        self._threshold_var = tk.DoubleVar(value=MAP_SCORE_THRESHOLD)
        self._threshold_display_var = tk.StringVar(
            value=f"{MAP_SCORE_THRESHOLD:.2f}"
        )
        self._min_file_kb_var = tk.IntVar(value=_MIN_FILE_BYTES // 1024)
        self._min_width_var = tk.IntVar(value=_MIN_IMAGE_WIDTH)
        self._min_height_var = tk.IntVar(value=_MIN_IMAGE_HEIGHT)
        self._t_low_var = tk.DoubleVar(value=DEFAULT_T_LOW)
        self._t_accept_var = tk.DoubleVar(value=DEFAULT_T_ACCEPT)
        self._t_reject_var = tk.DoubleVar(value=DEFAULT_T_REJECT)
        self._use_llm_gray_var = tk.BooleanVar(value=False)
        self._classification_var = tk.StringVar(value="")
        self._label_status_var = tk.StringVar(value="")
        self._t_low_display_var = tk.StringVar(value=f"{DEFAULT_T_LOW:.2f}")
        self._t_accept_display_var = tk.StringVar(value=f"{DEFAULT_T_ACCEPT:.2f}")
        self._t_reject_display_var = tk.StringVar(value=f"{DEFAULT_T_REJECT:.2f}")
        self._results_title_var = tk.StringVar(value="Найденные карты")
        self._results_hint_var = tk.StringVar(
            value="Список обновляется во время сканирования."
        )
        self._db_host_filter_var = tk.StringVar(value=_DB_FILTER_ALL)
        self._db_ml_filter_var = tk.StringVar(value=ML_FILTER_ALL)

        self._configure_styles()
        self._build_menu()
        self._build_ui()
        self._apply_theme()
        self._update_mode_summary()
        self.after(100, self._poll_events)
        self.after(200, self._maybe_load_database_on_startup)

    def _configure_styles(self) -> None:
        palette = _THEME_PALETTES[self._theme_name]
        style = ttk.Style(self)
        available = set(style.theme_names())
        if "clam" in available:
            style.theme_use("clam")

        base_bg = palette["base_bg"]
        card_bg = palette["card_bg"]
        accent = palette["accent"]
        text = palette["text"]
        muted = palette["muted"]
        border = palette["border"]

        style.configure("TFrame", background=base_bg)
        style.configure("Card.TFrame", background=card_bg)
        style.configure(
            "TLabelframe",
            background=card_bg,
            bordercolor=border,
            relief="solid",
        )
        style.configure(
            "TLabelframe.Label",
            background=card_bg,
            foreground=text,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "TLabel", background=base_bg, foreground=text, font=("Segoe UI", 10)
        )
        style.configure(
            "Card.TLabel", background=card_bg, foreground=text, font=("Segoe UI", 10)
        )
        style.configure(
            "Title.TLabel",
            background=base_bg,
            foreground=text,
            font=("Segoe UI Semibold", 18),
        )
        style.configure(
            "Subtitle.TLabel",
            background=base_bg,
            foreground=muted,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Section.TLabel",
            background=card_bg,
            foreground=text,
            font=("Segoe UI Semibold", 11),
        )
        style.configure(
            "MetricValue.TLabel",
            background=card_bg,
            foreground=accent,
            font=("Segoe UI Semibold", 20),
        )
        style.configure(
            "MetricCaption.TLabel",
            background=card_bg,
            foreground=muted,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Primary.TButton", padding=(14, 8), font=("Segoe UI Semibold", 10)
        )
        style.configure(
            "Secondary.TButton", padding=(12, 8), font=("Segoe UI", 10)
        )
        style.configure("TButton", padding=(10, 6), font=("Segoe UI", 10))
        style.configure("TNotebook", background=card_bg, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10))
        style.configure(
            "Modern.Horizontal.TProgressbar",
            troughcolor=border,
            background=accent,
            bordercolor=border,
            lightcolor=accent,
            darkcolor=accent,
        )
        style.configure(
            "TCheckbutton", background=card_bg, foreground=text, font=("Segoe UI", 10)
        )
        style.configure(
            "TRadiobutton", background=card_bg, foreground=text, font=("Segoe UI", 10)
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", card_bg), ("active", card_bg)],
            foreground=[("selected", text), ("active", text)],
        )
        style.map(
            "TCheckbutton",
            background=[("active", card_bg), ("selected", card_bg)],
            foreground=[("disabled", muted), ("active", text), ("selected", text)],
        )
        style.map(
            "TRadiobutton",
            background=[("active", card_bg), ("selected", card_bg)],
            foreground=[("disabled", muted), ("active", text), ("selected", text)],
        )

    def _current_palette(self) -> dict[str, str]:
        return _THEME_PALETTES[self._theme_name]

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Настройки", menu=settings_menu)
        settings_menu.add_command(
            label="Настройки...",
            command=self._open_search_settings,
            accelerator="Ctrl+,",
        )

        db_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="База данных", menu=db_menu)
        db_menu.add_command(
            label="Просмотр базы...",
            command=self._load_database_view,
            accelerator="Ctrl+B",
        )
        db_menu.add_separator()
        db_menu.add_command(
            label="Удалить выбранную запись...",
            command=self._delete_selected_from_db,
            accelerator="Delete",
        )
        db_menu.add_command(
            label="Очистить базу данных...",
            command=self._clear_database,
        )

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="О программе...", command=self._show_about)

        self.bind("<Control-comma>", lambda _e: self._open_search_settings())
        self.bind("<Control-b>", lambda _e: self._load_database_view())

    def _open_search_settings(self) -> None:
        if self._scan_thread and self._scan_thread.is_alive():
            messagebox.showinfo(
                APP_NAME,
                "Дождитесь завершения или остановите текущее сканирование.",
            )
            return
        if self._settings_dialog is not None and self._settings_dialog.winfo_exists():
            self._settings_dialog.lift()
            self._settings_dialog.focus_force()
            return
        if not self._remote_share_vars:
            defaults = set(default_remote_shares())
            for share in remote_share_choices():
                self._remote_share_vars[share] = tk.BooleanVar(
                    value=share in defaults
                )
        self._theme_var.set(self._theme_name)
        self._settings_dialog = SearchSettingsDialog(self)

    def _show_about(self) -> None:
        messagebox.showinfo(
            f"О программе — {APP_NAME}",
            f"{APP_NAME} {APP_VERSION}\n\n"
            f"{APP_DESCRIPTION}\n\n"
            f"Автор: {APP_AUTHOR}",
        )

    def _ollama_base_url(self) -> str:
        return normalize_ollama_base_url(self._llm_endpoint_var.get())

    def _detector_threshold(self) -> float:
        return float(self._threshold_var.get())

    def _hybrid_config(self) -> HybridConfig:
        return HybridConfig(
            heuristic_threshold=self._detector_threshold(),
            t_low=float(self._t_low_var.get()),
            t_accept=float(self._t_accept_var.get()),
            t_reject=float(self._t_reject_var.get()),
            use_llm_gray_zone=bool(self._use_llm_gray_var.get()),
        )

    def _size_filter_kwargs(self) -> dict[str, int]:
        return {
            "min_file_bytes": max(1, int(self._min_file_kb_var.get())) * 1024,
            "max_file_bytes": MAX_STORED_IMAGE_BYTES,
            "min_width": max(1, int(self._min_width_var.get())),
            "min_height": max(1, int(self._min_height_var.get())),
        }

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        shell = ttk.Frame(self, padding=16)
        shell.grid(row=0, column=0, sticky="nsew")
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        info_bar = ttk.Frame(shell, style="Card.TFrame", padding=14)
        info_bar.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        info_bar.columnconfigure(1, weight=1)

        mode_box = ttk.Frame(info_bar, style="Card.TFrame")
        mode_box.grid(row=0, column=0, sticky="w", padx=(0, 24))
        ttk.Label(
            mode_box, text="Активный режим", style="MetricCaption.TLabel"
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            mode_box, textvariable=self._mode_summary_var, style="Card.TLabel"
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        status_box = ttk.Frame(info_bar, style="Card.TFrame")
        status_box.grid(row=0, column=1, sticky="ew")
        status_box.columnconfigure(0, weight=1)
        ttk.Label(status_box, text="Статус", style="MetricCaption.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            status_box,
            textvariable=self._status_var,
            style="Card.TLabel",
            wraplength=900,
        ).grid(row=1, column=0, sticky="ew", pady=(4, 0))

        body = ttk.Frame(shell)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1, uniform="main")
        body.columnconfigure(1, weight=3, uniform="main")
        body.rowconfigure(0, weight=1)

        results_card = ttk.Frame(body, style="Card.TFrame", padding=16)
        results_card.grid(row=0, column=0, sticky="nsew")
        results_card.columnconfigure(0, weight=1)
        results_card.rowconfigure(2, weight=1)

        results_header = ttk.Frame(results_card, style="Card.TFrame")
        results_header.grid(row=0, column=0, sticky="ew")
        results_header.columnconfigure(0, weight=1)
        ttk.Label(
            results_header,
            textvariable=self._results_title_var,
            style="Section.TLabel",
        ).grid(row=0, column=0, sticky="w")
        self._open_db_button = ttk.Button(
            results_header,
            text="Открыть базу",
            command=self._load_database_view,
            style="Secondary.TButton",
        )
        self._open_db_button.grid(row=0, column=1, sticky="e", padx=(0, 8))
        self._delete_db_button = ttk.Button(
            results_header,
            text="Удалить из БД",
            command=self._delete_selected_from_db,
            style="Secondary.TButton",
        )
        self._delete_db_button.grid(row=0, column=2, sticky="e")
        ttk.Label(
            results_header,
            textvariable=self._results_hint_var,
            style="Card.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))

        self._db_filter_frame = ttk.Frame(results_card, style="Card.TFrame")
        self._db_filter_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self._db_filter_frame.columnconfigure(1, weight=1)
        ttk.Label(
            self._db_filter_frame,
            text="Компьютер",
            style="Card.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._db_host_combo = ttk.Combobox(
            self._db_filter_frame,
            textvariable=self._db_host_filter_var,
            state="normal",
            width=36,
        )
        self._db_host_combo.grid(row=0, column=1, sticky="ew")
        self._db_host_combo.bind("<<ComboboxSelected>>", self._on_db_host_filter_changed)
        self._db_host_combo.bind("<KeyRelease>", self._on_db_host_filter_changed)
        ttk.Label(
            self._db_filter_frame,
            text="выберите из списка или введите имя",
            style="Card.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ml_row = ttk.Frame(self._db_filter_frame, style="Card.TFrame")
        ml_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Label(ml_row, text="ML", style="Card.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        for column, (mode, label) in enumerate(_DB_ML_FILTER_LABELS.items()):
            ttk.Radiobutton(
                ml_row,
                text=label,
                value=mode,
                variable=self._db_ml_filter_var,
                command=self._apply_db_filters,
                style="TRadiobutton",
            ).grid(row=0, column=column + 1, sticky="w", padx=(0, 12))
        self._db_filter_frame.grid_remove()

        list_frame = ttk.Frame(results_card, style="Card.TFrame")
        list_frame.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self._listbox = tk.Listbox(
            list_frame,
            exportselection=False,
            activestyle="none",
            borderwidth=0,
            highlightthickness=0,
            font=("Segoe UI", 10),
            selectbackground="#dbeafe",
            selectforeground="#0f172a",
        )
        self._listbox.grid(row=0, column=0, sticky="nsew")
        self._listbox.bind("<<ListboxSelect>>", self._on_select)
        self._listbox.bind("<Double-Button-1>", self._on_listbox_double_click)
        self._listbox.bind("<Delete>", self._on_listbox_delete_key)

        yscroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._listbox.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self._listbox.configure(yscrollcommand=yscroll.set)

        preview_card = ttk.Frame(body, style="Card.TFrame", padding=16)
        preview_card.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        preview_card.columnconfigure(0, weight=1)
        preview_card.rowconfigure(1, weight=1)

        preview_header = ttk.Frame(preview_card, style="Card.TFrame")
        preview_header.grid(row=0, column=0, sticky="ew")
        preview_header.columnconfigure(0, weight=1)
        ttk.Label(
            preview_header,
            textvariable=self._selected_path_var,
            style="Card.TLabel",
            wraplength=680,
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="ew")
        ttk.Label(
            preview_header,
            textvariable=self._db_status_var,
            style="Card.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Label(
            preview_header,
            textvariable=self._classification_var,
            style="Card.TLabel",
        ).grid(row=2, column=0, sticky="w", pady=(4, 0))

        label_row = ttk.Frame(preview_header, style="Card.TFrame")
        label_row.grid(row=3, column=0, sticky="w", pady=(8, 0))
        self._label_map_button = ttk.Button(
            label_row,
            text="Это карта",
            command=lambda: self._save_user_label(1),
            style="Secondary.TButton",
        )
        self._label_map_button.grid(row=0, column=0, sticky="w")
        self._label_not_map_button = ttk.Button(
            label_row,
            text="Не карта",
            command=lambda: self._save_user_label(0),
            style="Secondary.TButton",
        )
        self._label_not_map_button.grid(row=0, column=1, padx=(8, 0), sticky="w")
        ttk.Label(
            label_row,
            textvariable=self._label_status_var,
            style="Card.TLabel",
        ).grid(row=0, column=2, padx=(12, 0), sticky="w")
        self._label_map_button.state(["disabled"])
        self._label_not_map_button.state(["disabled"])

        self._detail_tabs = ttk.Notebook(preview_card)
        self._detail_tabs.grid(row=1, column=0, sticky="nsew", pady=(12, 0))

        preview_tab = ttk.Frame(self._detail_tabs, style="Card.TFrame", padding=8)
        llm_tab = ttk.Frame(self._detail_tabs, style="Card.TFrame", padding=8)
        self._detail_tabs.add(preview_tab, text="Предпросмотр")
        self._detail_tabs.add(llm_tab, text="LLM-анализ")

        preview_tab.columnconfigure(0, weight=1)
        preview_tab.rowconfigure(0, weight=1)

        self._preview_label = ttk.Label(
            preview_tab,
            anchor=tk.CENTER,
            text="Нет выбранного файла.",
            style="Card.TLabel",
        )
        self._preview_label.grid(row=0, column=0, sticky="nsew")
        self._preview_label.bind("<Button-1>", self._on_preview_click)

        llm_tab.columnconfigure(0, weight=1)
        llm_tab.rowconfigure(2, weight=1)

        llm_controls = ttk.Frame(llm_tab, style="Card.TFrame")
        llm_controls.grid(row=0, column=0, sticky="ew")
        llm_controls.columnconfigure(1, weight=1)

        ttk.Label(
            llm_controls, text="Адрес Ollama:", style="Card.TLabel"
        ).grid(row=0, column=0, sticky="w")
        self._llm_endpoint_entry = ttk.Entry(
            llm_controls, textvariable=self._llm_endpoint_var, width=36
        )
        self._llm_endpoint_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self._check_ollama_button = ttk.Button(
            llm_controls,
            text="Проверить Ollama",
            command=self._check_ollama_connection,
            style="Secondary.TButton",
        )
        self._check_ollama_button.grid(row=0, column=2, padx=(10, 0), sticky="e")

        ttk.Label(
            llm_controls, text="Локальная модель:", style="Card.TLabel"
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))
        self._llm_model_entry = ttk.Entry(
            llm_controls, textvariable=self._llm_model_var, width=28
        )
        self._llm_model_entry.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))
        self._run_llm_button = ttk.Button(
            llm_controls,
            text="Запустить LLM-анализ",
            command=self._start_llm_analysis,
            style="Secondary.TButton",
        )
        self._run_llm_button.grid(row=1, column=2, padx=(10, 0), sticky="e", pady=(8, 0))
        self._run_llm_button.state(["disabled"])
        ttk.Label(
            llm_controls, textvariable=self._llm_status_var, style="Card.TLabel"
        ).grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        analysis_frame = ttk.Frame(llm_tab, style="Card.TFrame")
        analysis_frame.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        analysis_frame.columnconfigure(0, weight=1)
        analysis_frame.rowconfigure(0, weight=1)
        self._analysis_text = scrolledtext.ScrolledText(
            analysis_frame,
            wrap=tk.WORD,
            height=10,
            borderwidth=0,
            highlightthickness=0,
            font=("Segoe UI", 10),
        )
        self._analysis_text.grid(row=0, column=0, sticky="nsew")
        self._analysis_text.insert("1.0", "Подробный LLM-анализ будет показан здесь.")
        self._analysis_text.configure(state="disabled")

        footer = ttk.Frame(shell, style="Card.TFrame", padding=(14, 12))
        footer.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        footer.columnconfigure(2, weight=1)

        self._start_button = ttk.Button(
            footer,
            text="Начать сканирование",
            command=self._start_scan,
            style="Primary.TButton",
        )
        self._start_button.grid(row=0, column=0, sticky="w")
        self._stop_button = ttk.Button(
            footer,
            text="Остановить",
            command=self._stop_scan,
            style="Secondary.TButton",
        )
        self._stop_button.grid(row=0, column=1, padx=(10, 16), sticky="w")
        self._stop_button.state(["disabled"])

        self._progressbar = ttk.Progressbar(
            footer,
            mode="indeterminate",
            style="Modern.Horizontal.TProgressbar",
        )
        self._progressbar.grid(row=0, column=2, sticky="ew", padx=(0, 16))

        count_box = ttk.Frame(footer, style="Card.TFrame")
        count_box.grid(row=0, column=3, sticky="e")
        ttk.Label(
            count_box, text="Найдено карт", style="MetricCaption.TLabel"
        ).grid(row=0, column=0, sticky="e")
        ttk.Label(
            count_box, textvariable=self._count_var, style="MetricValue.TLabel"
        ).grid(row=0, column=1, sticky="e", padx=(8, 0))

    def _focus_llm_tab(self) -> None:
        if hasattr(self, "_detail_tabs"):
            self._detail_tabs.select(1)

    def _set_theme_from_settings(self) -> None:
        theme = self._theme_var.get()
        if theme not in _THEME_PALETTES:
            return
        self._theme_name = theme
        self._apply_theme()

    def _apply_theme(self) -> None:
        palette = _THEME_PALETTES[self._theme_name]
        self.configure(bg=palette["base_bg"])
        self._configure_styles()

        if hasattr(self, "_listbox"):
            self._listbox.configure(
                bg=palette["input_bg"],
                fg=palette["text"],
                selectbackground=palette["select_bg"],
                selectforeground=palette["select_fg"],
            )
        if hasattr(self, "_analysis_text"):
            self._analysis_text.configure(
                bg=palette["input_bg"],
                fg=palette["text"],
                insertbackground=palette["text"],
            )
        if (
            self._settings_dialog is not None
            and self._settings_dialog.winfo_exists()
        ):
            self._settings_dialog._apply_theme_from_parent()

    def _update_mode_summary(self) -> None:
        mode = self._scan_mode_var.get()
        threshold = self._detector_threshold()
        if mode == "all":
            source = "все локальные диски"
        elif mode == "drives":
            selected = sum(1 for var in self._drive_vars.values() if var.get())
            source = f"локальные диски, выбрано {selected}"
        elif mode == "folders":
            source = f"каталоги, добавлено {len(self._folder_paths)}"
        else:
            source = (
                f"компьютеры AD, выбрано {len(self._selected_ad_computers())}"
            )
        self._mode_summary_var.set(
            f"Источник: {source} | порог: {threshold:.2f}"
        )

    def _selected_remote_shares(self) -> list[str]:
        return [share for share, var in self._remote_share_vars.items() if var.get()]

    def _selected_ad_computers(self) -> list[str]:
        return [
            computer
            for computer in self._ad_computers
            if self._ad_computer_vars.get(computer) is not None
            and self._ad_computer_vars[computer].get()
        ]

    def _selected_roots(self) -> list[Path]:
        mode = self._scan_mode_var.get()
        if mode == "all":
            return _fixed_drive_roots()
        if mode == "drives":
            return [Path(path) for path, var in self._drive_vars.items() if var.get()]
        if mode == "folders":
            return [Path(path) for path in self._folder_paths]
        return build_unc_roots(self._selected_ad_computers(), self._selected_remote_shares())

    def _set_scan_running_state(self, is_running: bool) -> None:
        if is_running:
            self._start_button.state(["disabled"])
            self._stop_button.state(["!disabled"])
            self._progressbar.start(12)
        else:
            self._start_button.state(["!disabled"])
            self._stop_button.state(["disabled"])
            self._progressbar.stop()

    def _start_scan(self) -> None:
        if self._scan_thread and self._scan_thread.is_alive():
            return

        roots = self._selected_roots()
        scan_scope = self._scan_mode_var.get()
        if not roots:
            if scan_scope == "drives":
                text = "Выберите хотя бы один диск."
            elif scan_scope == "folders":
                text = "Добавьте хотя бы один каталог."
            elif scan_scope == "ad":
                selected_ad = self._selected_ad_computers()
                if not self._ad_computers:
                    text = "Сначала загрузите список компьютеров из AD."
                elif not selected_ad:
                    text = "Отметьте галочками хотя бы один компьютер для проверки."
                else:
                    text = "Выберите хотя бы одну удаленную шару, например C$."
            else:
                text = "Не найдено доступных дисков для сканирования."
            messagebox.showerror(APP_NAME, text)
            return

        threshold = self._detector_threshold()
        size_filters = self._size_filter_kwargs()
        hybrid_config = self._hybrid_config()
        use_llm_gray = bool(self._use_llm_gray_var.get())
        llm_model = self._llm_model_var.get().strip() or DEFAULT_LOCAL_MODEL
        llm_base_url = self._ollama_base_url()
        self._stop_requested.clear()
        self._viewing_database = False
        self._db_all_records.clear()
        self._set_db_filter_visible(False)
        self._results_title_var.set("Найденные карты")
        self._results_hint_var.set(
            "Список обновляется во время сканирования. "
            "Уже сохранённые карты не проверяются повторно."
        )
        self._found_paths.clear()
        self._found_image_ids.clear()
        self._selected_record = None
        self._listbox.delete(0, tk.END)
        self._preview_label.configure(image="", text="Нет выбранного файла.")
        self._preview_image = None
        self._preview_path = None
        self._preview_label.configure(cursor="")
        self._count_var.set("0")
        self._selected_path_var.set("Выберите файл из списка, чтобы посмотреть предпросмотр.")
        self._db_status_var.set(self._db_status_text())
        self._llm_status_var.set("LLM-анализ еще не выполнялся.")
        self._set_analysis_text("Подробный LLM-анализ будет показан здесь.")
        self._run_llm_button.state(["disabled"])
        self._status_var.set("Сканирование...")
        self._set_scan_running_state(True)

        self._scan_thread = threading.Thread(
            target=self._scan_worker,
            args=(roots, scan_scope, hybrid_config, size_filters, use_llm_gray, llm_model, llm_base_url),
            daemon=True,
        )
        self._scan_thread.start()

    def _stop_scan(self) -> None:
        if not self._scan_thread or not self._scan_thread.is_alive():
            return
        self._stop_requested.set()
        self._status_var.set("Остановка сканирования...")
        self._stop_button.state(["disabled"])

    def _scan_worker(
        self,
        roots: list[Path],
        scan_scope: str,
        hybrid_config: HybridConfig,
        size_filters: dict[str, int],
        use_llm_gray: bool,
        llm_model: str,
        llm_base_url: str,
    ) -> None:
        classified = 0
        maps = 0
        skipped = 0
        skipped_in_db = 0
        ml_classifier = self._ml_classifier
        path_index = self._image_store.build_path_index()

        def llm_classify(image_bytes: bytes) -> bool:
            return classify_map_yes_no(
                image_bytes=image_bytes,
                model_name=llm_model,
                base_url=llm_base_url,
            )

        llm_fn = llm_classify if use_llm_gray else None

        try:
            for root in roots:
                if self._stop_requested.is_set():
                    self._event_queue.put(
                        ("stopped", classified, maps, skipped, skipped_in_db)
                    )
                    return

                drive_files = 0
                skipped_here = 0
                self._event_queue.put(("status", f"Сканирование: {root}"))

                try:
                    if not root.exists() or not root.is_dir():
                        self._event_queue.put(("root_unavailable", str(root)))
                        continue
                except OSError as exc:
                    self._event_queue.put(("root_error", str(root), str(exc)))
                    continue

                try:
                    for fp in _walk_images(root):
                        if self._stop_requested.is_set():
                            self._event_queue.put(
                        ("stopped", classified, maps, skipped, skipped_in_db)
                    )
                            return
                        drive_files += 1
                        if not _passes_size_filters(fp, **size_filters):
                            skipped += 1
                            skipped_here += 1
                            continue
                        if lookup_path_in_index(fp, path_index) is not None:
                            skipped += 1
                            skipped_in_db += 1
                            skipped_here += 1
                            continue
                        decision = classify_image_path(
                            fp,
                            config=hybrid_config,
                            ml_classifier=ml_classifier,
                            llm_classify=llm_fn,
                        )
                        if decision is None:
                            continue
                        classified += 1
                        if decision.is_map:
                            maps += 1
                            try:
                                saved = self._image_store.save_detected_image(
                                    fp,
                                    scan_scope=scan_scope,
                                    heuristic=decision.as_heuristic_payload(),
                                )
                            except ImageTooLargeForStoreError:
                                skipped += 1
                                skipped_here += 1
                                continue
                            except Exception as exc:
                                self._event_queue.put(("db_error", str(fp), str(exc)))
                                continue
                            self._event_queue.put(
                                ("found", str(fp), saved.record.image_id, saved.inserted)
                            )
                            self._image_store.add_path_to_index(
                                path_index, fp, saved.record.image_id
                            )
                        if classified % _PROGRESS_EVERY == 0:
                            self._event_queue.put(
                                ("progress", classified, maps, skipped, skipped_in_db)
                            )
                except Exception as exc:
                    self._event_queue.put(("root_error", str(root), str(exc)))
                    continue

                self._event_queue.put(("root_done", str(root), drive_files, skipped_here))
            self._event_queue.put(("done", classified, maps, skipped, skipped_in_db))
        except Exception as exc:
            self._event_queue.put(("error", str(exc)))

    def _poll_events(self) -> None:
        try:
            while True:
                event = self._event_queue.get_nowait()
                kind = event[0]
                if kind == "status":
                    self._status_var.set(event[1])
                elif kind == "progress":
                    classified, maps, skipped = event[1], event[2], event[3]
                    skipped_in_db = int(event[4]) if len(event) > 4 else 0
                    self._status_var.set(
                        f"Проверено: {classified}, карт: {maps}, "
                        f"пропущено: {skipped} (уже в БД: {skipped_in_db})"
                    )
                elif kind == "root_done":
                    root, drive_files, skipped_here = event[1], event[2], event[3]
                    self._status_var.set(
                        f"Завершено {root}: файлов изображений {drive_files}, пропущено {skipped_here}"
                    )
                elif kind == "root_unavailable":
                    self._status_var.set(f"Недоступен путь: {event[1]}")
                elif kind == "root_error":
                    root, message = event[1], event[2]
                    self._status_var.set(f"Ошибка доступа {root}: {message}")
                elif kind == "db_error":
                    path, message = event[1], event[2]
                    self._status_var.set(f"Ошибка записи в БД для {path}: {message}")
                elif kind == "found":
                    path, image_id = event[1], int(event[2])
                    inserted = bool(event[3]) if len(event) > 3 else True
                    if path not in self._found_paths:
                        self._found_paths.append(path)
                        self._listbox.insert(tk.END, Path(path).name)
                        self._count_var.set(str(len(self._found_paths)))
                    self._found_image_ids[path] = image_id
                    if inserted:
                        self._db_status_var.set(
                            self._db_status_text(self._selected_record)
                        )
                    if not inserted:
                        self._status_var.set(
                            f"Карта уже в базе: {Path(path).name} (id={image_id})"
                        )
                elif kind == "done":
                    classified, maps, skipped = event[1], event[2], event[3]
                    skipped_in_db = int(event[4]) if len(event) > 4 else 0
                    self._write_report_csv()
                    self._status_var.set(
                        f"Готово. Проверено: {classified}, карт: {maps}, "
                        f"пропущено: {skipped} (уже в БД: {skipped_in_db})"
                    )
                    self._set_scan_running_state(False)
                elif kind == "stopped":
                    classified, maps, skipped = event[1], event[2], event[3]
                    skipped_in_db = int(event[4]) if len(event) > 4 else 0
                    self._write_report_csv()
                    self._status_var.set(
                        f"Остановлено. Проверено: {classified}, карт: {maps}, "
                        f"пропущено: {skipped} (уже в БД: {skipped_in_db})"
                    )
                    self._set_scan_running_state(False)
                elif kind == "error":
                    self._status_var.set(f"Ошибка: {event[1]}")
                    self._set_scan_running_state(False)
                    messagebox.showerror(APP_NAME, event[1])
                elif kind == "llm_started":
                    self._run_llm_button.state(["disabled"])
                    self._focus_llm_tab()
                    self._llm_status_var.set(
                        f"LLM-анализ запущен для модели {event[1]}..."
                    )
                elif kind == "llm_finished":
                    image_id = int(event[1])
                    self._run_llm_button.state(["!disabled"])
                    self._focus_llm_tab()
                    if self._selected_record and self._selected_record.image_id == image_id:
                        self._load_selected_record(image_id)
                    else:
                        self._llm_status_var.set("LLM-анализ завершен.")
                elif kind == "llm_failed":
                    image_id, message = int(event[1]), event[2]
                    self._run_llm_button.state(["!disabled"])
                    if self._selected_record and self._selected_record.image_id == image_id:
                        self._load_selected_record(image_id)
                    self._llm_status_var.set(f"Ошибка LLM-анализа: {message}")
                elif kind == "ollama_check":
                    self._check_ollama_button.state(["!disabled"])
                    self._llm_status_var.set(str(event[1]))
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _write_report_csv(self) -> None:
        out_path = Path.cwd() / _DEFAULT_OUTPUT
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["path"])
            for path in self._found_paths:
                writer.writerow([path])

    def _on_preview_click(self, _event: object) -> None:
        if self._preview_path is not None:
            self._open_image_file(self._preview_path)

    def _selected_list_path(self) -> Path | None:
        selection = self._listbox.curselection()
        if not selection:
            return None
        index = selection[0]
        if index < 0 or index >= len(self._found_paths):
            return None
        return Path(self._found_paths[index])

    def _on_listbox_double_click(self, _event: object) -> None:
        path = self._selected_list_path()
        if path is not None:
            self._open_image_file(path)

    @staticmethod
    def _open_image_file(path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_file():
            messagebox.showerror(
                APP_NAME,
                f"Файл не найден или недоступен:\n{resolved}",
            )
            return
        try:
            if sys.platform == "win32":
                os.startfile(resolved)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(resolved)], check=True)
            else:
                subprocess.run(["xdg-open", str(resolved)], check=True)
        except OSError as exc:
            messagebox.showerror(
                APP_NAME,
                f"Не удалось открыть файл:\n{resolved}\n\n{exc}",
            )
        except subprocess.CalledProcessError as exc:
            messagebox.showerror(
                APP_NAME,
                f"Не удалось открыть файл:\n{resolved}\n\n{exc}",
            )

    def _on_listbox_delete_key(self, _event: object) -> str:
        self._delete_selected_from_db()
        return "break"

    def _db_status_text(self, record: StoredImageRecord | None = None) -> str:
        store = self._image_store
        file_size = store.db_file_size_bytes()
        data_size = store.total_stored_image_bytes()
        base = (
            f"База данных: {store.db_path.name}, "
            f"записей: {store.count_images()}, "
            f"файл: {format_data_size(file_size)}, "
            f"данные: {format_data_size(data_size)}"
        )
        if record is None:
            return base
        return (
            f"{base} | id={record.image_id}, "
            f"{record.width}x{record.height}, {record.file_size // 1024} KiB"
        )

    def _classification_text(self, record: StoredImageRecord | None) -> str:
        if record is None:
            return ""
        summary = parse_feature_summary(record.feature_summary_json)
        ml_score = summary.get("ml_score")
        source = summary.get("decision_source") or "heuristic"
        verdict = "карта" if record.is_map else "не карта"
        heuristic_part = (
            f"heuristic={record.score:.2f}" if record.score is not None else "heuristic=—"
        )
        ml_part = f", ml={ml_score:.2f}" if ml_score is not None else ", ml=—"
        return f"{heuristic_part}{ml_part}, verdict={verdict}, source={source}"

    def _label_stats_text(self) -> str:
        maps, not_maps = self._image_store.get_label_stats()
        ready = "обучена" if self._ml_classifier.is_ready() else "не обучена"
        meta = self._ml_classifier.meta
        accuracy = ""
        if meta and meta.holdout_accuracy is not None:
            accuracy = f", hold-out: {meta.holdout_accuracy:.0%}"
        return (
            f"Метки: карт={maps}, не карт={not_maps}. "
            f"ML-модель: {ready}{accuracy}."
        )

    def _refresh_label_stats_in_settings(self) -> None:
        if (
            self._settings_dialog is not None
            and self._settings_dialog.winfo_exists()
        ):
            self._settings_dialog.refresh_label_stats()

    def _retrain_ml_classifier(self) -> str:
        samples = self._image_store.get_training_samples()
        result = self._ml_classifier.train(samples)
        self._refresh_label_stats_in_settings()
        return result.message

    def _save_user_label(self, label: int) -> None:
        if self._selected_record is None:
            messagebox.showinfo(APP_NAME, "Сначала выберите изображение из списка.")
            return
        record = self._selected_record
        try:
            features = self._image_store.get_features_for_image(record.image_id)
        except KeyError:
            messagebox.showerror(APP_NAME, "Запись не найдена в базе данных.")
            return
        self._image_store.save_user_label(
            sha256=record.sha256,
            label=label,
            features=features,
            source_path=record.source_path,
            image_id=record.image_id,
        )
        label_text = "карта" if label == 1 else "не карта"
        self._label_status_var.set(f"Размечено: {label_text}")
        self._refresh_label_stats_in_settings()
        self._status_var.set(f"Метка сохранена: {label_text} — {Path(record.source_path).name}")

    def _update_label_controls(self, record: StoredImageRecord | None) -> None:
        if record is None:
            self._label_map_button.state(["disabled"])
            self._label_not_map_button.state(["disabled"])
            self._label_status_var.set("")
            self._classification_var.set("")
            return
        self._label_map_button.state(["!disabled"])
        self._label_not_map_button.state(["!disabled"])
        self._classification_var.set(self._classification_text(record))
        user_label = self._image_store.get_user_label(record.sha256)
        if user_label is None:
            self._label_status_var.set("Метка не задана")
        else:
            self._label_status_var.set(
                "Размечено: карта" if user_label == 1 else "Размечено: не карта"
            )

    def _clear_preview_panel(self) -> None:
        self._selected_record = None
        self._preview_path = None
        self._preview_image = None
        self._preview_label.configure(
            image="", text="Нет выбранного файла.", cursor=""
        )
        self._selected_path_var.set(
            "Выберите файл из списка, чтобы посмотреть предпросмотр."
        )
        self._db_status_var.set(self._db_status_text())
        self._update_label_controls(None)
        self._llm_status_var.set("LLM-анализ еще не выполнялся.")
        self._set_analysis_text("Подробный LLM-анализ будет показан здесь.")
        self._run_llm_button.state(["disabled"])

    def _remove_list_item_at(self, index: int) -> None:
        if index < 0 or index >= len(self._found_paths):
            return
        path = self._found_paths.pop(index)
        self._found_image_ids.pop(str(path), None)
        self._listbox.delete(index)
        self._count_var.set(str(len(self._found_paths)))

    def _set_db_filter_visible(self, visible: bool) -> None:
        if visible:
            self._db_filter_frame.grid()
        else:
            self._db_filter_frame.grid_remove()
            self._db_host_filter_var.set(_DB_FILTER_ALL)
            self._db_ml_filter_var.set(ML_FILTER_ALL)

    def _format_db_list_item(
        self,
        record: StoredImageRecord,
        *,
        show_host: bool,
    ) -> str:
        name = Path(record.source_path).name
        ml = record_ml_score(record)
        ml_part = f"  ml={ml * 100:.0f}%" if ml is not None else ""
        line = f"{name}  [#{record.image_id}]{ml_part}"
        if show_host:
            host = record.source_host or _DB_FILTER_LOCAL_LABEL
            return f"{host}: {line}"
        return line

    def _db_records_for_current_host_filter(self) -> list[StoredImageRecord]:
        text = self._db_host_filter_var.get().strip()
        if not text or text == _DB_FILTER_ALL:
            return list(self._db_all_records)

        if text == _DB_FILTER_LOCAL_LABEL or text.startswith(
            f"{_DB_FILTER_LOCAL_LABEL} ("
        ):
            return [record for record in self._db_all_records if not record.source_host]

        match = re.fullmatch(r"(.+?) \(\d+\)", text)
        if match:
            host = match.group(1)
            if host == _DB_FILTER_LOCAL_LABEL:
                return [
                    record for record in self._db_all_records if not record.source_host
                ]
            host_lower = host.lower()
            return [
                record
                for record in self._db_all_records
                if record.source_host and record.source_host.lower() == host_lower
            ]

        needle = text.lower()
        return [
            record
            for record in self._db_all_records
            if record.source_host and needle in record.source_host.lower()
        ]

    def _db_records_for_current_ml_filter(
        self,
        records: list[StoredImageRecord],
    ) -> list[StoredImageRecord]:
        mode = self._db_ml_filter_var.get().strip()
        if mode in ("", ML_FILTER_ALL):
            return records
        return [
            record
            for record in records
            if matches_ml_score_filter(record_ml_score(record), filter_mode=mode)
        ]

    def _db_records_for_current_filters(self) -> list[StoredImageRecord]:
        records = self._db_records_for_current_host_filter()
        return self._db_records_for_current_ml_filter(records)

    def _active_db_filter_labels(self) -> list[str]:
        labels: list[str] = []
        host_text = self._db_host_filter_var.get().strip()
        if host_text and host_text != _DB_FILTER_ALL:
            labels.append(host_text)
        ml_mode = self._db_ml_filter_var.get().strip()
        if ml_mode and ml_mode != ML_FILTER_ALL:
            labels.append(_DB_ML_FILTER_LABELS.get(ml_mode, ml_mode))
        return labels

    def _rebuild_db_host_filter_choices(self) -> None:
        host_counts: dict[str, int] = {}
        local_count = 0
        for record in self._db_all_records:
            if record.source_host:
                host_counts[record.source_host] = host_counts.get(record.source_host, 0) + 1
            else:
                local_count += 1

        choices = [_DB_FILTER_ALL]
        if local_count:
            choices.append(f"{_DB_FILTER_LOCAL_LABEL} ({local_count})")
        for host in sorted(host_counts, key=str.lower):
            choices.append(f"{host} ({host_counts[host]})")
        self._db_host_combo.configure(values=choices)

    def _refresh_db_listbox(self, records: list[StoredImageRecord]) -> None:
        show_host = self._db_host_filter_var.get().strip() in ("", _DB_FILTER_ALL)
        self._found_paths.clear()
        self._found_image_ids.clear()
        self._listbox.delete(0, tk.END)
        for record in records:
            path = record.source_path
            self._found_paths.append(path)
            self._listbox.insert(
                tk.END,
                self._format_db_list_item(record, show_host=show_host),
            )
            self._found_image_ids[path] = record.image_id

        total = len(self._db_all_records)
        shown = len(records)
        if shown == total:
            self._count_var.set(str(total))
        else:
            self._count_var.set(f"{shown} / {total}")

    def _apply_db_filters(self) -> None:
        if not self._viewing_database:
            return
        records = self._db_records_for_current_filters()
        self._refresh_db_listbox(records)
        active = self._active_db_filter_labels()
        if not active:
            self._status_var.set(f"База данных: {len(self._db_all_records)} карт.")
        else:
            joined = ", ".join(active)
            self._status_var.set(
                f"Отбор ({joined}): {len(records)} из {len(self._db_all_records)} карт."
            )

    def _on_db_host_filter_changed(self, _event: object = None) -> None:
        self._apply_db_filters()

    def _maybe_load_database_on_startup(self) -> None:
        if self._scan_thread and self._scan_thread.is_alive():
            return
        if self._image_store.count_images() > 0:
            self._load_database_view(silent=True)

    def _load_database_view(self, *, silent: bool = False) -> None:
        if self._scan_thread and self._scan_thread.is_alive():
            messagebox.showinfo(
                APP_NAME,
                "Дождитесь завершения или остановите текущее сканирование.",
            )
            return

        try:
            records = self._image_store.list_image_records()
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Не удалось загрузить базу данных:\n{exc}")
            return

        if not records:
            if not silent:
                messagebox.showinfo(APP_NAME, "База данных пуста.")
            return

        self._viewing_database = True
        self._results_title_var.set("База данных")
        self._results_hint_var.set(
            "Сохранённые карты без повторного сканирования. "
            "Фильтры: компьютер и ML-оценка."
        )
        self._db_all_records = records
        self._selected_record = None
        self._clear_preview_panel()
        self._set_db_filter_visible(True)
        self._rebuild_db_host_filter_choices()
        self._db_host_filter_var.set(_DB_FILTER_ALL)
        self._db_ml_filter_var.set(ML_FILTER_ALL)
        self._apply_db_filters()
        self._db_status_var.set(self._db_status_text())

    def _delete_selected_from_db(self) -> None:
        if self._scan_thread and self._scan_thread.is_alive():
            messagebox.showinfo(
                APP_NAME,
                "Дождитесь завершения или остановите текущее сканирование.",
            )
            return

        selection = self._listbox.curselection()
        if not selection:
            messagebox.showinfo(APP_NAME, "Выберите запись в списке найденных карт.")
            return

        index = selection[0]
        if index < 0 or index >= len(self._found_paths):
            return

        path = Path(self._found_paths[index])
        image_id = self._found_image_ids.get(str(path))
        if image_id is None:
            messagebox.showinfo(
                APP_NAME,
                "Для выбранного файла нет записи в базе данных.",
            )
            return

        if not messagebox.askyesno(
            APP_NAME,
            f"Удалить запись из базы данных?\n\n"
            f"Файл: {path.name}\n"
            f"id: {image_id}\n\n"
            f"Файл на диске не удаляется.",
        ):
            return

        deleted = self._image_store.delete_image(image_id)
        if not deleted:
            messagebox.showerror(APP_NAME, f"Запись id={image_id} не найдена в базе.")
            return

        deleted_id = self._selected_record.image_id if self._selected_record else None
        if self._viewing_database:
            self._db_all_records = [
                record
                for record in self._db_all_records
                if record.image_id != image_id
            ]
            if not self._db_all_records:
                self._viewing_database = False
                self._set_db_filter_visible(False)
                self._found_paths.clear()
                self._found_image_ids.clear()
                self._listbox.delete(0, tk.END)
                self._count_var.set("0")
            else:
                self._rebuild_db_host_filter_choices()
                self._apply_db_filters()
        else:
            self._remove_list_item_at(index)
        if deleted_id == image_id:
            self._clear_preview_panel()
        else:
            self._db_status_var.set(self._db_status_text(self._selected_record))

        self._status_var.set(f"Запись id={image_id} удалена из базы данных.")

    def _clear_database(self) -> None:
        if self._scan_thread and self._scan_thread.is_alive():
            messagebox.showinfo(
                APP_NAME,
                "Дождитесь завершения или остановите текущее сканирование.",
            )
            return

        total = self._image_store.count_images()
        if total == 0:
            messagebox.showinfo(APP_NAME, "База данных уже пуста.")
            return

        if not messagebox.askyesno(
            APP_NAME,
            f"Удалить все записи из базы данных ({total})?\n\n"
            f"Файлы на диске не удаляются.",
        ):
            return

        removed = self._image_store.clear_all()
        self._viewing_database = False
        self._db_all_records.clear()
        self._set_db_filter_visible(False)
        self._results_title_var.set("Найденные карты")
        self._results_hint_var.set(
            "Список обновляется во время сканирования. "
            "Уже сохранённые карты не проверяются повторно."
        )
        self._found_paths.clear()
        self._found_image_ids.clear()
        self._listbox.delete(0, tk.END)
        self._count_var.set("0")
        self._clear_preview_panel()
        self._status_var.set(f"База данных очищена. Удалено записей: {removed}.")

    def _on_select(self, _event: object) -> None:
        path = self._selected_list_path()
        if path is None:
            return
        self._selected_path_var.set(str(path))
        image_id = self._found_image_ids.get(str(path))
        self._show_preview(path, image_id=image_id)
        if image_id is not None:
            self._load_selected_record(image_id)

    def _load_selected_record(self, image_id: int) -> None:
        try:
            record = self._image_store.get_image_record(image_id)
        except KeyError:
            self._found_image_ids = {
                path: item_id
                for path, item_id in self._found_image_ids.items()
                if item_id != image_id
            }
            self._db_status_var.set(self._db_status_text())
            self._llm_status_var.set("Запись удалена из базы данных.")
            return
        self._selected_record = record
        self._db_status_var.set(self._db_status_text(record))
        self._update_label_controls(record)
        llm_status = record.llm_status or "нет подробного анализа"
        self._llm_status_var.set(
            f"LLM-статус: {llm_status}"
            + (f", модель: {record.llm_model_name}" if record.llm_model_name else "")
        )
        text = record.llm_analysis_text or (
            "Подробный LLM-анализ еще не выполнялся.\n\n"
            f"Heuristic score: {record.score}\n"
            f"Features: {record.feature_summary_json}"
        )
        self._set_analysis_text(text)
        self._run_llm_button.state(["!disabled"])

    def _set_analysis_text(self, text: str) -> None:
        self._analysis_text.configure(state="normal")
        self._analysis_text.delete("1.0", tk.END)
        self._analysis_text.insert("1.0", text)
        self._analysis_text.configure(state="disabled")

    def _check_ollama_connection(self) -> None:
        base_url = self._ollama_base_url()
        model_name = self._llm_model_var.get().strip() or DEFAULT_LOCAL_MODEL
        self._llm_endpoint_var.set(base_url)
        self._check_ollama_button.state(["disabled"])
        self._llm_status_var.set("Проверка Ollama...")

        def worker() -> None:
            health = ollama_health_check(base_url)
            message = format_health_check_message(health, model_name=model_name)
            self._event_queue.put(("ollama_check", message))

        threading.Thread(target=worker, daemon=True).start()

    def _start_llm_analysis(self) -> None:
        if self._selected_record is None:
            messagebox.showinfo(APP_NAME, "Сначала выберите изображение из списка.")
            return
        model_name = self._llm_model_var.get().strip() or DEFAULT_LOCAL_MODEL
        base_url = self._ollama_base_url()
        self._llm_endpoint_var.set(base_url)
        self._run_llm_button.state(["disabled"])
        thread = threading.Thread(
            target=self._run_llm_analysis_worker,
            args=(self._selected_record.image_id, model_name, base_url),
            daemon=True,
        )
        thread.start()

    def _run_llm_analysis_worker(
        self, image_id: int, model_name: str, base_url: str
    ) -> None:
        self._event_queue.put(("llm_started", model_name))
        try:
            record = self._image_store.get_image_record(image_id)
            image_bytes = self._image_store.get_image_bytes(image_id)
            self._image_store.update_llm_result(
                image_id,
                status="running",
                model_name=model_name,
                prompt_version=None,
                analysis_text=None,
                structured_json=None,
            )
            result = analyze_image_with_local_llm(
                image_bytes=image_bytes,
                record=record,
                model_name=model_name,
                base_url=base_url,
            )
            self._image_store.update_llm_result(
                image_id,
                status=str(result["status"]),
                model_name=str(result["model_name"]),
                prompt_version=str(result["prompt_version"]),
                analysis_text=str(result["analysis_text"]),
                structured_json=result.get("structured_json"),
            )
            self._event_queue.put(("llm_finished", image_id))
        except Exception as exc:
            self._image_store.update_llm_result(
                image_id,
                status="failed",
                model_name=model_name,
                prompt_version=None,
                analysis_text=str(exc),
                structured_json=None,
            )
            self._event_queue.put(("llm_failed", image_id, str(exc)))

    def _show_preview(self, path: Path, *, image_id: int | None = None) -> None:
        self._preview_path = path
        img = imread_unicode(path)
        preview_from_db = False
        if (img is None or img.size == 0) and image_id is not None:
            try:
                data = self._image_store.get_image_bytes(image_id)
                buf = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
                preview_from_db = img is not None and img.size > 0
            except (KeyError, OSError):
                img = None

        if img is None or img.size == 0:
            self._preview_image = None
            self._preview_path = None
            self._preview_label.configure(
                image="",
                text="Не удалось открыть изображение.",
                cursor="",
            )
            return

        h, w = img.shape[:2]
        scale = min(_PREVIEW_MAX_W / max(w, 1), _PREVIEW_MAX_H / max(h, 1), 1.0)
        if scale < 1.0:
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(".png", img)
        if not ok:
            self._preview_image = None
            self._preview_path = None
            self._preview_label.configure(
                image="", text="Не удалось подготовить предпросмотр.", cursor=""
            )
            return

        data_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        self._preview_image = tk.PhotoImage(data=data_b64, format="png")
        cursor = "" if preview_from_db else "hand2"
        self._preview_path = None if preview_from_db else path
        self._preview_label.configure(image=self._preview_image, text="", cursor=cursor)


def main() -> None:
    app = MapScannerGui()
    app.mainloop()


if __name__ == "__main__":
    main()
