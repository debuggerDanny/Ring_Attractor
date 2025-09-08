#!/usr/bin/env python3
"""
Visual test results viewer for Ring Attractor project.

This script helps open test results in a browser and provides
a simple GUI for running tests with visual feedback.
"""

import webbrowser
import os
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
from pathlib import Path
import time


class TestRunner:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ring Attractor Test Runner")
        self.root.geometry("800x600")
        
        # Test results directory
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.setup_ui()
        self.running = False
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Ring Attractor Test Runner", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Test type selection
        test_frame = ttk.LabelFrame(main_frame, text="Test Selection", padding="10")
        test_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.test_type = tk.StringVar(value="all")
        
        ttk.Radiobutton(test_frame, text="All Tests", variable=self.test_type, 
                       value="all").grid(row=0, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Radiobutton(test_frame, text="Unit Tests", variable=self.test_type, 
                       value="unit").grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        ttk.Radiobutton(test_frame, text="Integration Tests", variable=self.test_type, 
                       value="integration").grid(row=0, column=2, sticky=tk.W)
        
        # Options
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.coverage = tk.BooleanVar(value=True)
        self.verbose = tk.BooleanVar()
        self.fast = tk.BooleanVar()
        
        ttk.Checkbutton(options_frame, text="Coverage Report", 
                       variable=self.coverage).grid(row=1, column=0, sticky=tk.W, padx=(0, 20))
        ttk.Checkbutton(options_frame, text="Verbose Output", 
                       variable=self.verbose).grid(row=1, column=1, sticky=tk.W, padx=(0, 20))
        ttk.Checkbutton(options_frame, text="Fast (Skip Slow Tests)", 
                       variable=self.fast).grid(row=1, column=2, sticky=tk.W)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=(20, 10))
        
        self.run_button = ttk.Button(button_frame, text="Run Tests", 
                                    command=self.run_tests, style="Accent.TButton")
        self.run_button.grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(button_frame, text="View Coverage Report", 
                  command=self.view_coverage).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Button(button_frame, text="Open Results Folder", 
                  command=self.open_results_folder).grid(row=0, column=2, padx=(0, 10))
        
        ttk.Button(button_frame, text="Clear Results", 
                  command=self.clear_results).grid(row=0, column=3)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Output text area
        output_frame = ttk.LabelFrame(main_frame, text="Test Output", padding="5")
        output_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, 
                                                    height=20, font=("Consolas", 9))
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready to run tests")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def log_output(self, message):
        """Add message to output text area"""
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)
        self.root.update_idletasks()
    
    def run_tests(self):
        """Run tests in a separate thread"""
        if self.running:
            messagebox.showwarning("Tests Running", "Tests are already running!")
            return
        
        # Clear output
        self.output_text.delete(1.0, tk.END)
        
        # Start progress bar
        self.progress.start(10)
        self.run_button.config(state="disabled")
        self.running = True
        self.status_var.set("Running tests...")
        
        # Run tests in thread
        thread = threading.Thread(target=self._run_tests_thread)
        thread.daemon = True
        thread.start()
    
    def _run_tests_thread(self):
        """Run tests in background thread"""
        try:
            # Build command
            cmd = [sys.executable, "run_tests.py"]
            
            if self.test_type.get() == "unit":
                cmd.append("--unit")
            elif self.test_type.get() == "integration":
                cmd.append("--integration")
            else:
                cmd.append("--all")
            
            if self.coverage.get():
                cmd.extend(["--coverage", "--html-report"])
            
            if self.verbose.get():
                cmd.append("--verbose")
            
            if self.fast.get():
                cmd.append("--fast")
            
            # Add JUnit XML for parsing
            cmd.append("--junit-xml")
            
            self.log_output(f"Running command: {' '.join(cmd)}")
            self.log_output("=" * 60)
            
            # Execute command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output
            for line in iter(process.stdout.readline, ''):
                self.log_output(line.rstrip())
            
            process.wait()
            
            # Update status
            if process.returncode == 0:
                self.status_var.set("Tests completed successfully!")
                self.log_output("\n[SUCCESS] All tests passed!")
                if self.coverage.get():
                    self.log_output(f"Coverage report: {self.results_dir}/htmlcov/index.html")
            else:
                self.status_var.set("Tests failed!")
                self.log_output("\n[FAILED] Some tests failed!")
        
        except Exception as e:
            self.log_output(f"\nError running tests: {e}")
            self.status_var.set("Error running tests")
        
        finally:
            # Stop progress and re-enable button
            self.progress.stop()
            self.run_button.config(state="normal")
            self.running = False
    
    def view_coverage(self):
        """Open coverage report in browser"""
        coverage_file = self.results_dir / "htmlcov" / "index.html"
        if coverage_file.exists():
            webbrowser.open(f"file://{coverage_file.absolute()}")
        else:
            messagebox.showinfo("Coverage Report", 
                               "No coverage report found. Run tests with coverage option first.")
    
    def open_results_folder(self):
        """Open results folder in file explorer"""
        if sys.platform == "win32":
            os.startfile(self.results_dir)
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", str(self.results_dir)])
        else:  # Linux
            subprocess.run(["xdg-open", str(self.results_dir)])
    
    def clear_results(self):
        """Clear test results"""
        try:
            import shutil
            if self.results_dir.exists():
                shutil.rmtree(self.results_dir)
                self.results_dir.mkdir(exist_ok=True)
            
            # Clear output
            self.output_text.delete(1.0, tk.END)
            self.status_var.set("Results cleared")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear results: {e}")
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()


def open_coverage_report():
    """Quick function to open coverage report"""
    coverage_file = Path("test_results/htmlcov/index.html")
    if coverage_file.exists():
        webbrowser.open(f"file://{coverage_file.absolute()}")
        print(f"Opened coverage report: {coverage_file}")
    else:
        print("No coverage report found. Run tests with --coverage --html-report first.")


def open_junit_report():
    """Quick function to open JUnit XML report"""
    junit_file = Path("test_results/junit.xml")
    if junit_file.exists():
        print(f"JUnit report: {junit_file}")
        print("You can view this in CI/CD systems or XML viewers")
    else:
        print("No JUnit report found. Run tests with --junit-xml first.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "coverage":
            open_coverage_report()
        elif sys.argv[1] == "junit":
            open_junit_report()
        else:
            print("Usage:")
            print("  python view_results.py          # Open GUI")
            print("  python view_results.py coverage # Open coverage report")
            print("  python view_results.py junit    # Show JUnit report")
    else:
        # Start GUI
        app = TestRunner()
        app.run()