"""
Tkinter GUI for Manga Extract-and-Clean.
"""

import os
import platform
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import backend


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Manga Extract & Clean")
        self.root.geometry("700x480")
        self.root.resizable(True, True)

        self._output_folder: str = ""
        self._processing = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # === Process Section ===
        process_frame = ttk.LabelFrame(self.root, text="1. Process Manga Folder")
        process_frame.pack(fill="x", **pad)

        row = ttk.Frame(process_frame)
        row.pack(fill="x", **pad)
        ttk.Label(row, text="Manga Folder:").pack(side="left")
        self.folder_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.folder_var, width=50).pack(side="left", fill="x", expand=True, padx=(5, 5))
        ttk.Button(row, text="Browse...", command=self._browse_folder).pack(side="left")

        btn_row = ttk.Frame(process_frame)
        btn_row.pack(fill="x", **pad)
        self.process_btn = ttk.Button(btn_row, text="Process", command=self._on_process)
        self.process_btn.pack(side="left")

        self.progress_bar = ttk.Progressbar(process_frame, mode="determinate")
        self.progress_bar.pack(fill="x", **pad)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(process_frame, textvariable=self.status_var, anchor="w").pack(fill="x", **pad)

        # === Output Section ===
        output_frame = ttk.LabelFrame(self.root, text="Output")
        output_frame.pack(fill="x", **pad)

        row2 = ttk.Frame(output_frame)
        row2.pack(fill="x", **pad)
        ttk.Label(row2, text="Output Folder:").pack(side="left")
        self.output_var = tk.StringVar(value="(not yet processed)")
        ttk.Label(row2, textvariable=self.output_var, anchor="w", foreground="blue").pack(side="left", fill="x", expand=True, padx=(5, 5))
        self.open_folder_btn = ttk.Button(row2, text="Open Folder", command=self._open_output_folder, state="disabled")
        self.open_folder_btn.pack(side="left")

        # === Script to JSON Section ===
        script_frame = ttk.LabelFrame(self.root, text="2. Turn Translated Script to JSON")
        script_frame.pack(fill="x", **pad)

        ttk.Label(script_frame, text="After translating lmt_script.txt with an LLM, select the translated file below:").pack(anchor="w", **pad)

        row3 = ttk.Frame(script_frame)
        row3.pack(fill="x", **pad)
        ttk.Label(row3, text="Translated Script:").pack(side="left")
        self.translated_var = tk.StringVar()
        ttk.Entry(row3, textvariable=self.translated_var, width=50).pack(side="left", fill="x", expand=True, padx=(5, 5))
        ttk.Button(row3, text="Browse...", command=self._browse_translated).pack(side="left")

        btn_row2 = ttk.Frame(script_frame)
        btn_row2.pack(fill="x", **pad)
        self.script_to_json_btn = ttk.Button(btn_row2, text="Convert to JSON", command=self._on_script_to_json)
        self.script_to_json_btn.pack(side="left")

        self.s2j_status_var = tk.StringVar(value="")
        ttk.Label(script_frame, textvariable=self.s2j_status_var, anchor="w").pack(fill="x", **pad)

    # ------------------------------------------------------------------
    # Browse dialogs
    # ------------------------------------------------------------------

    def _browse_folder(self):
        path = filedialog.askdirectory(title="Select Manga Folder")
        if path:
            self.folder_var.set(path)

    def _browse_translated(self):
        path = filedialog.askopenfilename(
            title="Select Translated Script",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            self.translated_var.set(path)

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def _on_process(self):
        folder = self.folder_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Invalid Folder", "Please select a valid manga folder.")
            return
        if self._processing:
            return

        self._processing = True
        self.process_btn.config(state="disabled")
        self.progress_bar["value"] = 0
        self.status_var.set("Starting...")

        output_folder = os.path.join(folder, "output")
        self._output_folder = output_folder

        thread = threading.Thread(
            target=self._process_worker,
            args=(folder, output_folder),
            daemon=True,
        )
        thread.start()

    def _process_worker(self, input_folder: str, output_folder: str):
        try:
            failures = backend.process_folder(
                input_folder, output_folder, self._progress_callback
            )
            self.root.after(0, self._process_done, failures)
        except Exception as e:
            self.root.after(0, self._process_error, str(e))

    def _progress_callback(self, stage: str, current: int, total: int):
        self.root.after(0, self._update_progress, stage, current, total)

    def _update_progress(self, stage: str, current: int, total: int):
        if total > 0:
            self.progress_bar["value"] = (current / total) * 100
        self.status_var.set(f"{stage}  ({current}/{total})")

    def _process_done(self, failures: list):
        self._processing = False
        self.process_btn.config(state="normal")
        self.progress_bar["value"] = 100
        self.output_var.set(self._output_folder)
        self.open_folder_btn.config(state="normal")

        if failures:
            names = "\n".join(f"  - {name}: {err}" for name, err in failures)
            messagebox.showwarning(
                "Completed with errors",
                f"Processing finished but {len(failures)} image(s) failed:\n{names}",
            )
            self.status_var.set(f"Done ({len(failures)} error(s))")
        else:
            self.status_var.set("Done — all images processed successfully")

    def _process_error(self, error_msg: str):
        self._processing = False
        self.process_btn.config(state="normal")
        self.status_var.set("Error")
        messagebox.showerror("Processing Error", error_msg)

    # ------------------------------------------------------------------
    # Script to JSON conversion
    # ------------------------------------------------------------------

    def _on_script_to_json(self):
        translated = self.translated_var.get().strip()
        if not translated or not os.path.isfile(translated):
            messagebox.showwarning("Invalid File", "Please select a valid translated script file.")
            return

        if not self._output_folder or not os.path.isdir(self._output_folder):
            messagebox.showwarning(
                "No Output Folder",
                "Please process a manga folder first so that lmt_raw.json exists.",
            )
            return

        original_path = os.path.join(self._output_folder, "lmt_raw.json")
        if not os.path.isfile(original_path):
            messagebox.showerror(
                "Missing File",
                f"Cannot find {original_path}.\nPlease process a manga folder first.",
            )
            return

        try:
            out_path, count, warnings = backend.script_to_json(
                original_path, translated, self._output_folder
            )
            msg = f"Converted successfully!\nInjected {count} translations.\nSaved to: {out_path}"
            if warnings:
                msg += "\n\nWarnings:\n" + "\n".join(warnings)
            self.s2j_status_var.set(f"Done — {count} translations injected → {os.path.basename(out_path)}")
            messagebox.showinfo("Conversion Complete", msg)
        except Exception as e:
            self.s2j_status_var.set("Error")
            messagebox.showerror("Conversion Error", str(e))

    # ------------------------------------------------------------------
    # Open output folder
    # ------------------------------------------------------------------

    def _open_output_folder(self):
        folder = self._output_folder
        if not folder or not os.path.isdir(folder):
            messagebox.showinfo("No Output", "Output folder does not exist yet.")
            return
        if platform.system() == "Windows":
            os.startfile(folder)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])

