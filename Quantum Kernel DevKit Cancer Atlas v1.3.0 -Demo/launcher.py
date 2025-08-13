import sys, os, time, socket, threading, webbrowser
from pathlib import Path
from contextlib import contextmanager
import logging

LOG_NAME = "devkit_launch.log"

@contextmanager
def check_single_instance(cwd: Path):
    """Ensure only one instance runs using a lock file."""
    lock_file = cwd / "devkit.lock"
    logger = logging.getLogger("launcher")
    if lock_file.exists():
        logger.error("Another instance of DevKit is running.")
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, "Another instance of DevKit is running.", "DevKit", 0x10)
        except Exception:
            pass
        sys.exit(3)
    lock_file.touch()
    try:
        yield
    finally:
        try:
            lock_file.unlink()
        except Exception:
            logger.error("Failed to remove lock file.")

def resource_roots():
    """Return (ROOT, CWD) that work for both frozen and dev runs."""
    if getattr(sys, "frozen", False):
        ROOT = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        CWD = Path(sys.executable).parent
    else:
        ROOT = Path(__file__).resolve().parent
        CWD = ROOT
    return ROOT, CWD

def find_app(root: Path, cwd: Path) -> Path | None:
    """Locate app.py next to the EXE (or in MEIPASS)."""
    for p in (cwd / "app.py", root / "app.py"):
        if p.exists():
            return p
    return None

def find_free_port(start=8501, tries=15):
    """Find an available port starting from 'start'."""
    for p in range(start, start + tries):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return start

def wait_for_port(host: str, port: int, timeout: float = 60.0) -> bool:
    """Poll the TCP port until it accepts connections or timeout."""
    deadline = time.time() + timeout
    logger = logging.getLogger("launcher")
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(1.0)
            try:
                s.connect((host, port))
                return True
            except OSError:
                time.sleep(0.4)
    logger.error(f"Port {port} not reachable after {timeout} seconds.")
    return False

def msgbox(title, text):
    """Show a Windows message box."""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, text, title, 0x10)
    except Exception:
        pass

def main():
    # Setup logging
    logging.basicConfig(filename=LOG_NAME, level=logging.DEBUG, filemode="w", encoding="utf-8")
    logger = logging.getLogger("launcher")

    ROOT, CWD = resource_roots()
    logger.info(f"sys.executable={sys.executable}")
    logger.info(f"ROOT={ROOT}")
    logger.info(f"CWD={CWD}")

    with check_single_instance(CWD):
        APP = find_app(ROOT, CWD)
        if not APP:
            logger.error("app.py not found.")
            msgbox("DevKit", "Couldn't find app.py next to the EXE.")
            return 1
        logger.info(f"Using APP={APP}")

        # Environment & port
        os.chdir(CWD)
        os.environ["PYTHONIOENCODING"] = "utf-8"
        os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        port = find_free_port(8501, 15)
        logger.info(f"Chosen port={port}")

        # Run Streamlit in-process
        from streamlit.web import cli as stcli

        def run_server():
            try:
                sys.argv = [
                    "streamlit", "run", str(APP),
                    "--server.port", str(port),
                    "--server.headless", "true",
                    "--logger.level", "debug"
                ]
                logger.info(f"Starting Streamlit: {' '.join(sys.argv)}")
                stcli.main()
            except Exception as e:
                logger.error(f"Streamlit failed: {e}", exc_info=True)
                msgbox("DevKit", f"Streamlit server failed: {e}")
                sys.exit(2)

        t = threading.Thread(target=run_server, daemon=True)
        t.start()

        # Wait for server to be ready
        if wait_for_port("127.0.0.1", port, timeout=60):
            logger.info(f"Server ready on http://localhost:{port}")
            webbrowser.open_new(f"http://localhost:{port}")
            while t.is_alive():
                time.sleep(0.5)
            return 0
        else:
            logger.error("Streamlit server didn't start within 60 seconds.")
            msgbox("DevKit", "Streamlit server didn't start. See devkit_launch.log.")
            return 2

if __name__ == "__main__":
    sys.exit(main())