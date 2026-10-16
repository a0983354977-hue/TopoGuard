"""Build a native standalone download for the current operating system."""

import platform
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
DOWNLOADS_DIR = PROJECT_DIR / "downloads"


def main() -> None:
    if sys.platform == "win32":
        platform_name = "Windows"
    elif sys.platform == "linux":
        platform_name = "Linux"
    elif sys.platform == "darwin":
        platform_name = "macOS"
    else:
        raise RuntimeError(f"Unsupported build platform: {sys.platform}")

    output_dir = DOWNLOADS_DIR / platform_name
    work_dir = PROJECT_DIR / "build" / platform_name
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        "TopoGuard",
        "--collect-binaries",
        "gudhi",
        "--distpath",
        str(output_dir),
        "--workpath",
        str(work_dir),
    ]

    if platform_name in {"Windows", "macOS"}:
        command.append("--windowed")
    if platform_name == "Windows":
        icon_path = PROJECT_DIR / "assets" / "topoguard.ico"
        if icon_path.is_file():
            command.extend(["--icon", str(icon_path)])

    command.append(str(PROJECT_DIR / "tda_guard_gui.py"))
    subprocess.run(command, check=True, cwd=PROJECT_DIR)

    if platform_name == "Linux":
        executable = output_dir / "TopoGuard"
        archive = output_dir / f"TopoGuard-linux-{platform.machine().lower()}.tar.gz"
        executable.chmod(0o755)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(executable, arcname=executable.name)
        print(f"Linux download created: {archive}")
    elif platform_name == "macOS":
        app_bundle = output_dir / "TopoGuard.app"
        archive = output_dir / f"TopoGuard-macos-{platform.machine().lower()}.zip"
        if not app_bundle.is_dir():
            raise FileNotFoundError(f"PyInstaller did not create the expected app bundle: {app_bundle}")
        ditto = shutil.which("ditto")
        if ditto is None:
            raise RuntimeError("The macOS 'ditto' utility is required to package the app bundle.")
        subprocess.run(
            [ditto, "-c", "-k", "--sequesterRsrc", "--keepParent", str(app_bundle), str(archive)],
            check=True,
            cwd=PROJECT_DIR,
        )
        print(f"macOS download created: {archive}")
    else:
        print(f"Windows download created: {output_dir / 'TopoGuard.exe'}")


if __name__ == "__main__":
    main()
