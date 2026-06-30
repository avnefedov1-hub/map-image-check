"""
Modal dialog for scan source and detector tuning parameters.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import TYPE_CHECKING

from .ad_remote import (
    AdComputerRecord,
    check_computers_online,
    default_remote_shares,
    list_enabled_ad_computer_records,
    remote_share_choices,
)
from .app_info import APP_NAME
from .detector import MAP_SCORE_THRESHOLD
from .hybrid_pipeline import DEFAULT_T_ACCEPT, DEFAULT_T_LOW, DEFAULT_T_REJECT
from .scan_drives import (
    _MIN_FILE_BYTES,
    _MIN_IMAGE_HEIGHT,
    _MIN_IMAGE_WIDTH,
)

if TYPE_CHECKING:
    from .gui_app import MapScannerGui

_MODE_TABS = (
    ("all", "Все диски"),
    ("drives", "Диски"),
    ("folders", "Каталоги"),
    ("ad", "Компьютеры AD"),
)


class SearchSettingsDialog(tk.Toplevel):
    """Separate window for scan sources and detector accuracy settings."""

    def __init__(self, parent: MapScannerGui) -> None:
        super().__init__(parent)
        self._app = parent
        self.title("Настройки")
        self.geometry("720x680")
        self.minsize(640, 520)
        self.transient(parent)
        self.grab_set()

        self._tab_by_mode: dict[str, ttk.Frame] = {}
        self._mode_by_tab_id: dict[str, str] = {}
        self._ad_online_check_running = False
        self._ad_online_progress_var = tk.StringVar(value="")

        self._build_ui()
        self._select_mode_tab(self._app._scan_mode_var.get())
        self._apply_theme_from_parent()
        self.refresh_label_stats()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<Escape>", lambda _e: self._on_close())

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        header = ttk.Frame(root)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(
            header,
            text="Настройки",
            style="Section.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Поиск, точность детектора и оформление интерфейса.",
            style="Card.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        notebook = ttk.Notebook(root)
        notebook.grid(row=1, column=0, sticky="nsew", pady=(12, 12))
        notebook.bind("<<NotebookTabChanged>>", self._on_source_tab_changed)
        self._source_tabs = notebook

        all_frame = ttk.Frame(notebook, style="Card.TFrame", padding=12)
        drives_frame = ttk.Frame(notebook, style="Card.TFrame", padding=8)
        folders_frame = ttk.Frame(notebook, style="Card.TFrame", padding=8)
        ad_frame = ttk.Frame(notebook, style="Card.TFrame", padding=8)
        detector_frame = ttk.Frame(notebook, style="Card.TFrame", padding=12)
        appearance_frame = ttk.Frame(notebook, style="Card.TFrame", padding=12)

        self._tab_by_mode = {
            "all": all_frame,
            "drives": drives_frame,
            "folders": folders_frame,
            "ad": ad_frame,
        }
        for mode, title in _MODE_TABS:
            tab = self._tab_by_mode[mode]
            notebook.add(tab, text=title)
            self._mode_by_tab_id[str(tab)] = mode

        notebook.add(detector_frame, text="Точность")
        notebook.add(appearance_frame, text="Оформление")

        ttk.Label(
            all_frame,
            text="Будут просканированы все доступные локальные диски.",
            style="Card.TLabel",
            wraplength=560,
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            all_frame,
            text="Подходит для полного локального поиска карт на текущем компьютере.",
            style="Card.TLabel",
            wraplength=560,
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        self._drives_frame = drives_frame
        self._folders_frame = folders_frame
        self._ad_frame = ad_frame
        self._build_drive_controls()
        self._build_folder_controls()
        self._build_ad_controls()
        self._build_detector_controls(detector_frame)
        self._build_appearance_controls(appearance_frame)

        buttons = ttk.Frame(root)
        buttons.grid(row=2, column=0, sticky="e")
        ttk.Button(
            buttons,
            text="Закрыть",
            command=self._on_close,
            style="Primary.TButton",
        ).grid(row=0, column=0)

    def _build_appearance_controls(self, frame: ttk.Frame) -> None:
        app = self._app
        ttk.Label(
            frame,
            text="Тема интерфейса",
            style="Card.TLabel",
            wraplength=560,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        theme_row = ttk.Frame(frame, style="Card.TFrame")
        theme_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Radiobutton(
            theme_row,
            text="Светлая",
            variable=app._theme_var,
            value="light",
            command=app._set_theme_from_settings,
        ).grid(row=0, column=0, padx=(0, 16), sticky="w")
        ttk.Radiobutton(
            theme_row,
            text="Тёмная",
            variable=app._theme_var,
            value="dark",
            command=app._set_theme_from_settings,
        ).grid(row=0, column=1, sticky="w")

        ttk.Label(
            frame,
            text="Изменения применяются сразу.",
            style="Card.TLabel",
            wraplength=560,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))

    def _build_detector_controls(self, frame: ttk.Frame) -> None:
        app = self._app
        frame.columnconfigure(1, weight=1)

        ttk.Label(
            frame,
            text="Настройки гибридного пайплайна: эвристика → ML → LLM (серая зона).",
            style="Card.TLabel",
            wraplength=560,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(frame, text="Порог «похоже на карту»:", style="Card.TLabel").grid(
            row=1, column=0, sticky="w", pady=(14, 4)
        )
        threshold_row = ttk.Frame(frame, style="Card.TFrame")
        threshold_row.grid(row=1, column=1, sticky="ew", pady=(14, 4))
        threshold_row.columnconfigure(0, weight=1)
        ttk.Scale(
            threshold_row,
            from_=0.20,
            to=0.80,
            orient=tk.HORIZONTAL,
            variable=app._threshold_var,
            command=self._on_threshold_scale,
        ).grid(row=0, column=0, sticky="ew")
        ttk.Label(
            threshold_row,
            textvariable=app._threshold_display_var,
            style="Card.TLabel",
            width=6,
        ).grid(row=0, column=1, padx=(8, 0))

        ttk.Label(
            frame,
            text="Меньше — больше находок (больше ложных срабатываний). "
            "Больше — строже отбор.",
            style="Card.TLabel",
            wraplength=560,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(frame, text="Мин. размер файла (КиБ):", style="Card.TLabel").grid(
            row=3, column=0, sticky="w", pady=(4, 4)
        )
        ttk.Spinbox(
            frame,
            from_=10,
            to=1024,
            increment=10,
            textvariable=app._min_file_kb_var,
            width=10,
        ).grid(row=3, column=1, sticky="w", pady=(4, 4))

        ttk.Label(frame, text="Мин. ширина (px):", style="Card.TLabel").grid(
            row=4, column=0, sticky="w", pady=(4, 4)
        )
        ttk.Spinbox(
            frame,
            from_=50,
            to=2000,
            increment=50,
            textvariable=app._min_width_var,
            width=10,
        ).grid(row=4, column=1, sticky="w", pady=(4, 4))

        ttk.Label(frame, text="Мин. высота (px):", style="Card.TLabel").grid(
            row=5, column=0, sticky="w", pady=(4, 4)
        )
        ttk.Spinbox(
            frame,
            from_=50,
            to=2000,
            increment=50,
            textvariable=app._min_height_var,
            width=10,
        ).grid(row=5, column=1, sticky="w", pady=(4, 4))

        ttk.Label(
            frame,
            text="Файлы меньше указанных размеров пропускаются без проверки. "
            "Исходные файлы больше 50 МиБ не обрабатываются. "
            "В базу сохраняется JPEG с уменьшенным разрешением (до 2048 px по длинной стороне).",
            style="Card.TLabel",
            wraplength=560,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Separator(frame).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(16, 8))

        ttk.Label(frame, text="T_low (быстрый отсев):", style="Card.TLabel").grid(
            row=8, column=0, sticky="w", pady=(4, 4)
        )
        t_low_row = ttk.Frame(frame, style="Card.TFrame")
        t_low_row.grid(row=8, column=1, sticky="ew", pady=(4, 4))
        t_low_row.columnconfigure(0, weight=1)
        ttk.Scale(
            t_low_row,
            from_=0.10,
            to=0.60,
            orient=tk.HORIZONTAL,
            variable=app._t_low_var,
            command=lambda v: app._t_low_display_var.set(f"{float(v):.2f}"),
        ).grid(row=0, column=0, sticky="ew")
        ttk.Label(
            t_low_row,
            textvariable=app._t_low_display_var,
            style="Card.TLabel",
            width=6,
        ).grid(row=0, column=1, padx=(8, 0))

        ttk.Label(frame, text="T_accept (ML → карта):", style="Card.TLabel").grid(
            row=9, column=0, sticky="w", pady=(4, 4)
        )
        t_accept_row = ttk.Frame(frame, style="Card.TFrame")
        t_accept_row.grid(row=9, column=1, sticky="ew", pady=(4, 4))
        t_accept_row.columnconfigure(0, weight=1)
        ttk.Scale(
            t_accept_row,
            from_=0.50,
            to=0.95,
            orient=tk.HORIZONTAL,
            variable=app._t_accept_var,
            command=lambda v: app._t_accept_display_var.set(f"{float(v):.2f}"),
        ).grid(row=0, column=0, sticky="ew")
        ttk.Label(
            t_accept_row,
            textvariable=app._t_accept_display_var,
            style="Card.TLabel",
            width=6,
        ).grid(row=0, column=1, padx=(8, 0))

        ttk.Label(frame, text="T_reject (ML → не карта):", style="Card.TLabel").grid(
            row=10, column=0, sticky="w", pady=(4, 4)
        )
        t_reject_row = ttk.Frame(frame, style="Card.TFrame")
        t_reject_row.grid(row=10, column=1, sticky="ew", pady=(4, 4))
        t_reject_row.columnconfigure(0, weight=1)
        ttk.Scale(
            t_reject_row,
            from_=0.05,
            to=0.50,
            orient=tk.HORIZONTAL,
            variable=app._t_reject_var,
            command=lambda v: app._t_reject_display_var.set(f"{float(v):.2f}"),
        ).grid(row=0, column=0, sticky="ew")
        ttk.Label(
            t_reject_row,
            textvariable=app._t_reject_display_var,
            style="Card.TLabel",
            width=6,
        ).grid(row=0, column=1, padx=(8, 0))

        ttk.Checkbutton(
            frame,
            text="LLM для сомнительных (серая зона ML, медленно)",
            variable=app._use_llm_gray_var,
        ).grid(row=11, column=0, columnspan=2, sticky="w", pady=(12, 4))

        self._label_stats_var = tk.StringVar(value=app._label_stats_text())
        ttk.Label(
            frame,
            textvariable=self._label_stats_var,
            style="Card.TLabel",
            wraplength=560,
        ).grid(row=12, column=0, columnspan=2, sticky="w", pady=(8, 4))

        train_row = ttk.Frame(frame, style="Card.TFrame")
        train_row.grid(row=13, column=0, columnspan=2, sticky="w", pady=(4, 0))
        ttk.Button(
            train_row,
            text="Переобучить модель",
            command=self._retrain_model,
            style="Secondary.TButton",
        ).grid(row=0, column=0, sticky="w")

        ttk.Button(
            frame,
            text="Сбросить точность",
            command=self._reset_detector_defaults,
            style="Secondary.TButton",
        ).grid(row=14, column=0, columnspan=2, sticky="w", pady=(16, 0))

    def refresh_label_stats(self) -> None:
        self._label_stats_var.set(self._app._label_stats_text())

    def _retrain_model(self) -> None:
        message = self._app._retrain_ml_classifier()
        self.refresh_label_stats()
        if self._app._ml_classifier.is_ready():
            messagebox.showinfo(APP_NAME, message)
        else:
            messagebox.showwarning(APP_NAME, message)

    def _on_threshold_scale(self, value: str) -> None:
        self._app._threshold_display_var.set(f"{float(value):.2f}")

    def _reset_detector_defaults(self) -> None:
        app = self._app
        app._threshold_var.set(MAP_SCORE_THRESHOLD)
        app._threshold_display_var.set(f"{MAP_SCORE_THRESHOLD:.2f}")
        app._min_file_kb_var.set(_MIN_FILE_BYTES // 1024)
        app._min_width_var.set(_MIN_IMAGE_WIDTH)
        app._min_height_var.set(_MIN_IMAGE_HEIGHT)
        app._t_low_var.set(DEFAULT_T_LOW)
        app._t_accept_var.set(DEFAULT_T_ACCEPT)
        app._t_reject_var.set(DEFAULT_T_REJECT)
        app._t_low_display_var.set(f"{DEFAULT_T_LOW:.2f}")
        app._t_accept_display_var.set(f"{DEFAULT_T_ACCEPT:.2f}")
        app._t_reject_display_var.set(f"{DEFAULT_T_REJECT:.2f}")
        app._use_llm_gray_var.set(False)
        app._update_mode_summary()

    def _build_drive_controls(self) -> None:
        from .scan_drives import _fixed_drive_roots

        for child in self._drives_frame.winfo_children():
            child.destroy()

        self._drives_frame.columnconfigure(0, weight=1)
        ttk.Label(
            self._drives_frame,
            text="Выберите локальные диски для сканирования.",
            style="Card.TLabel",
            wraplength=560,
        ).grid(row=0, column=0, sticky="w")

        drives = _fixed_drive_roots()
        checks = ttk.Frame(self._drives_frame, style="Card.TFrame")
        checks.grid(row=1, column=0, sticky="w", pady=(4, 0))

        for index, drive in enumerate(drives):
            key = str(drive)
            if key not in self._app._drive_vars:
                self._app._drive_vars[key] = tk.BooleanVar(value=False)
            var = self._app._drive_vars[key]
            ttk.Checkbutton(
                checks,
                text=key,
                variable=var,
                command=self._app._update_mode_summary,
            ).grid(
                row=index // 4,
                column=index % 4,
                padx=(0, 12),
                pady=(0, 8),
                sticky="w",
            )

        if not drives:
            ttk.Label(
                self._drives_frame,
                text="Доступные диски не найдены.",
                style="Card.TLabel",
            ).grid(row=2, column=0, sticky="w")

    def _build_folder_controls(self) -> None:
        app = self._app
        for child in self._folders_frame.winfo_children():
            child.destroy()

        self._folders_frame.columnconfigure(0, weight=1)
        self._folders_frame.rowconfigure(2, weight=1)
        ttk.Label(
            self._folders_frame,
            text="Добавьте один или несколько каталогов для выборочного сканирования.",
            style="Card.TLabel",
            wraplength=560,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        buttons = ttk.Frame(self._folders_frame, style="Card.TFrame")
        buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 10))
        ttk.Button(buttons, text="Добавить каталог", command=self._add_folder).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(buttons, text="Убрать", command=self._remove_folder).grid(
            row=0, column=1, padx=(8, 0), sticky="w"
        )

        list_frame = ttk.Frame(self._folders_frame, style="Card.TFrame")
        list_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(4, 0))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self._folders_listbox = tk.Listbox(
            list_frame,
            height=10,
            exportselection=False,
            activestyle="none",
            borderwidth=0,
            highlightthickness=0,
            font=("Segoe UI", 10),
        )
        self._folders_listbox.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self._folders_listbox.yview
        )
        scroll.grid(row=0, column=1, sticky="ns")
        self._folders_listbox.configure(yscrollcommand=scroll.set)

        for path in app._folder_paths:
            self._folders_listbox.insert(tk.END, path)
        self._apply_list_theme(self._folders_listbox)

    def _build_ad_controls(self) -> None:
        app = self._app
        for child in self._ad_frame.winfo_children():
            child.destroy()

        self._ad_frame.columnconfigure(0, weight=3)
        self._ad_frame.columnconfigure(1, weight=2)
        self._ad_frame.rowconfigure(3, weight=1)
        ttk.Label(
            self._ad_frame,
            text=(
                "Сначала загрузите все компьютеры из AD, "
                "затем «Проверить включение» — галочки только у online рабочих станций "
                "(серверы помечены и не отмечаются автоматически)."
            ),
            style="Card.TLabel",
            wraplength=560,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        action_row = ttk.Frame(self._ad_frame, style="Card.TFrame")
        action_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 10))
        action_row.columnconfigure(5, weight=1)

        ttk.Button(action_row, text="Загрузить из AD", command=self._load_ad_computers).grid(
            row=0, column=0, sticky="w"
        )
        self._check_ad_online_button = ttk.Button(
            action_row,
            text="Проверить включение",
            command=self._check_ad_computers_online,
        )
        self._check_ad_online_button.grid(row=0, column=1, padx=(8, 0), sticky="w")
        ttk.Button(
            action_row, text="Выбрать все", command=self._select_all_ad_computers
        ).grid(row=0, column=2, padx=(8, 0), sticky="w")
        ttk.Button(
            action_row, text="Снять все", command=self._clear_ad_selection
        ).grid(row=0, column=3, padx=(8, 0), sticky="w")
        ttk.Button(
            action_row, text="Очистить список", command=self._clear_ad_computers
        ).grid(row=0, column=4, padx=(8, 0), sticky="w")
        ttk.Label(
            action_row,
            textvariable=app._ad_status_var,
            style="Card.TLabel",
            wraplength=360,
        ).grid(row=0, column=5, padx=(12, 0), sticky="w")

        progress_row = ttk.Frame(self._ad_frame, style="Card.TFrame")
        progress_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        progress_row.columnconfigure(1, weight=1)
        ttk.Label(
            progress_row,
            textvariable=self._ad_online_progress_var,
            style="Card.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=(0, 12))
        self._ad_online_progressbar = ttk.Progressbar(
            progress_row,
            mode="determinate",
            maximum=100,
        )
        self._ad_online_progressbar.grid(row=0, column=1, sticky="ew")

        list_frame = ttk.LabelFrame(self._ad_frame, text="Компьютеры", padding=6)
        list_frame.grid(row=3, column=0, sticky="nsew", padx=(0, 8))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        palette = self._app._current_palette()
        self._ad_canvas = tk.Canvas(
            list_frame,
            highlightthickness=0,
            height=220,
            background=palette["card_bg"],
        )
        self._ad_canvas.grid(row=0, column=0, sticky="nsew")
        ad_scroll = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL, command=self._ad_canvas.yview
        )
        ad_scroll.grid(row=0, column=1, sticky="ns")
        self._ad_canvas.configure(yscrollcommand=ad_scroll.set)

        self._ad_checks_frame = ttk.Frame(self._ad_canvas, style="Card.TFrame")
        self._ad_canvas_window = self._ad_canvas.create_window(
            (0, 0), window=self._ad_checks_frame, anchor="nw"
        )
        self._ad_checks_frame.bind("<Configure>", self._on_ad_checks_configure)
        self._ad_canvas.bind("<Configure>", self._on_ad_canvas_configure)
        self._refresh_ad_checkbuttons()

        shares_frame = ttk.LabelFrame(self._ad_frame, text="Удалённые шары", padding=6)
        shares_frame.grid(row=3, column=1, sticky="nsew")
        shares_frame.columnconfigure(0, weight=1)

        ttk.Label(
            shares_frame,
            text="Выберите диски для проверки на удалённых компьютерах.",
            style="Card.TLabel",
            wraplength=220,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 8))

        if not app._remote_share_vars:
            defaults = set(default_remote_shares())
            for share in remote_share_choices():
                app._remote_share_vars[share] = tk.BooleanVar(value=share in defaults)

        for index, share in enumerate(remote_share_choices()):
            var = app._remote_share_vars[share]
            ttk.Checkbutton(shares_frame, text=share, variable=var).grid(
                row=1 + index // 4,
                column=index % 4,
                sticky="w",
                padx=(0, 12),
                pady=(0, 6),
            )

    def _apply_list_theme(self, listbox: tk.Listbox) -> None:
        palette = self._app._current_palette()
        listbox.configure(
            bg=palette["input_bg"],
            fg=palette["text"],
            selectbackground=palette["select_bg"],
            selectforeground=palette["select_fg"],
        )

    def _apply_theme_from_parent(self) -> None:
        palette = self._app._current_palette()
        self.configure(bg=palette["base_bg"])
        if hasattr(self, "_folders_listbox"):
            self._apply_list_theme(self._folders_listbox)
        if hasattr(self, "_ad_canvas"):
            self._ad_canvas.configure(
                background=palette["card_bg"],
                highlightbackground=palette["border"],
            )

    def _select_mode_tab(self, mode: str) -> None:
        tab = self._tab_by_mode[mode]
        self._source_tabs.select(tab)
        self._app._scan_mode_var.set(mode)
        self._app._update_mode_summary()

    def _on_source_tab_changed(self, _event: object) -> None:
        current_tab_id = self._source_tabs.select()
        widget_name = str(self.nametowidget(current_tab_id)) if current_tab_id else ""
        mode = self._mode_by_tab_id.get(widget_name)
        if mode:
            self._app._scan_mode_var.set(mode)
            if mode == "drives":
                self._build_drive_controls()
            self._app._update_mode_summary()

    def _add_folder(self) -> None:
        selected = filedialog.askdirectory(
            title="Выберите каталог для сканирования", parent=self
        )
        if not selected:
            return
        path = str(Path(selected).resolve())
        if path in self._app._folder_paths:
            return
        self._app._folder_paths.append(path)
        self._folders_listbox.insert(tk.END, path)
        self._app._update_mode_summary()

    def _remove_folder(self) -> None:
        selection = self._folders_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        del self._app._folder_paths[index]
        self._folders_listbox.delete(index)
        self._app._update_mode_summary()

    def _load_ad_computers(self) -> None:
        app = self._app
        app._ad_status_var.set("Загрузка компьютеров из AD...")
        self.update_idletasks()

        def worker() -> None:
            try:
                records = list_enabled_ad_computer_records()
                app.after(0, lambda: self._apply_ad_computers(records))
            except Exception as exc:
                app.after(0, lambda: self._ad_load_failed(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _check_ad_computers_online(self) -> None:
        app = self._app
        if self._ad_online_check_running:
            return
        if not app._ad_computers:
            messagebox.showinfo(
                "Map Image Check",
                "Сначала загрузите список компьютеров из AD.",
                parent=self,
            )
            return

        computers = list(app._ad_computers)
        total = len(computers)
        self._ad_online_check_running = True
        self._check_ad_online_button.state(["disabled"])
        self._start_ad_online_progress(total)

        def worker() -> None:
            def on_progress(done: int, count: int, name: str | None) -> None:
                app.after(0, lambda: self._update_ad_online_progress(done, count, name))

            try:
                online = check_computers_online(computers, progress_callback=on_progress)
                app.after(0, lambda: self._apply_ad_online_status(online))
            except Exception as exc:
                app.after(0, lambda: self._ad_online_check_failed(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _start_ad_online_progress(self, total: int) -> None:
        self._ad_online_progress_var.set(f"Проверка включения: 0 из {total}")
        self._ad_online_progressbar.configure(value=0, maximum=max(total, 1))
        app = self._app
        app._ad_status_var.set(f"Проверка включения {total} компьютеров...")
        self.update_idletasks()

    def _update_ad_online_progress(
        self, done: int, total: int, computer_name: str | None
    ) -> None:
        total = max(total, 1)
        self._ad_online_progressbar.configure(value=done, maximum=total)
        name_part = f" — {computer_name}" if computer_name else ""
        self._ad_online_progress_var.set(
            f"Проверка включения: {done} из {total}{name_part}"
        )
        self._app._ad_status_var.set(
            f"Проверка включения: {done} из {total}{name_part}"
        )

    def _finish_ad_online_progress(self) -> None:
        self._ad_online_check_running = False
        self._check_ad_online_button.state(["!disabled"])
        self._ad_online_progress_var.set("")
        self._ad_online_progressbar.configure(value=0)

    def _apply_ad_computers(self, records: list[AdComputerRecord]) -> None:
        app = self._app
        app._ad_computers = [record.name for record in records]
        app._ad_computer_is_server = {
            record.name: bool(record.is_server) for record in records
        }
        app._ad_computer_vars = {
            record.name: tk.BooleanVar(value=False) for record in records
        }
        self._refresh_ad_checkbuttons()
        if records:
            server_count = sum(1 for record in records if record.is_server)
            workstation_count = len(records) - server_count
            app._ad_status_var.set(
                f"Загружено: {len(records)} "
                f"(рабочие станции: {workstation_count}, серверы: {server_count}). "
                "Нажмите «Проверить включение» для online рабочих станций."
            )
            app._update_mode_summary()
        else:
            app._ad_status_var.set("В Active Directory не найдено включённых компьютеров.")

    def _apply_ad_online_status(self, online: dict[str, bool]) -> None:
        app = self._app
        for computer in app._ad_computers:
            var = app._ad_computer_vars.get(computer)
            if var is None:
                var = tk.BooleanVar(value=False)
                app._ad_computer_vars[computer] = var
            is_server = app._ad_computer_is_server.get(computer, False)
            var.set(bool(online.get(computer, False)) and not is_server)
        self._refresh_ad_checkbuttons()
        online_count = sum(
            1 for computer in app._ad_computers if online.get(computer, False)
        )
        selected_count = sum(
            1
            for computer in app._ad_computers
            if online.get(computer, False)
            and not app._ad_computer_is_server.get(computer, False)
        )
        server_count = sum(1 for computer in app._ad_computers if app._ad_computer_is_server.get(computer, False))
        selected = len(app._selected_ad_computers())
        app._ad_status_var.set(
            f"Загружено: {len(app._ad_computers)}, online: {online_count}, "
            f"отмечено рабочих станций: {selected_count}, серверов: {server_count}, "
            f"выбрано для проверки: {selected}"
        )
        app._update_mode_summary()
        self._finish_ad_online_progress()

    def _ad_online_check_failed(self, exc: Exception) -> None:
        self._finish_ad_online_progress()
        self._app._ad_status_var.set("Не удалось проверить включение компьютеров.")
        messagebox.showerror("Map Image Check", str(exc), parent=self)

    def _ad_load_failed(self, exc: Exception) -> None:
        self._app._ad_status_var.set("Не удалось загрузить список компьютеров AD.")
        messagebox.showerror("Map Image Check", str(exc), parent=self)

    def _clear_ad_computers(self) -> None:
        self._app._ad_computers.clear()
        self._app._ad_computer_vars.clear()
        self._app._ad_computer_is_server.clear()
        self._refresh_ad_checkbuttons()
        self._app._ad_status_var.set("Список компьютеров очищен.")

    def _refresh_ad_checkbuttons(self) -> None:
        for child in self._ad_checks_frame.winfo_children():
            child.destroy()

        for row, computer in enumerate(self._app._ad_computers):
            var = self._app._ad_computer_vars.get(computer)
            if var is None:
                var = tk.BooleanVar(value=False)
                self._app._ad_computer_vars[computer] = var
            label = computer
            if self._app._ad_computer_is_server.get(computer, False):
                label = f"{computer} (сервер)"
            ttk.Checkbutton(
                self._ad_checks_frame,
                text=label,
                variable=var,
                command=self._update_ad_status,
            ).grid(row=row, column=0, sticky="w")

        self._ad_checks_frame.update_idletasks()
        self._ad_canvas.configure(scrollregion=self._ad_canvas.bbox("all"))

    def _select_all_ad_computers(self) -> None:
        for var in self._app._ad_computer_vars.values():
            var.set(True)
        self._update_ad_status()

    def _clear_ad_selection(self) -> None:
        for var in self._app._ad_computer_vars.values():
            var.set(False)
        self._update_ad_status()

    def _update_ad_status(self) -> None:
        app = self._app
        total = len(app._ad_computers)
        selected = len(app._selected_ad_computers())
        app._ad_status_var.set(
            f"Загружено компьютеров: {total}, выбрано для проверки: {selected}"
        )
        app._update_mode_summary()

    def _on_ad_checks_configure(self, _event: object) -> None:
        self._ad_canvas.configure(scrollregion=self._ad_canvas.bbox("all"))

    def _on_ad_canvas_configure(self, event: object) -> None:
        width = getattr(event, "width", None)
        if width:
            self._ad_canvas.itemconfigure(self._ad_canvas_window, width=width)

    def _on_close(self) -> None:
        self._app._settings_dialog = None
        self._app._update_mode_summary()
        self.destroy()
