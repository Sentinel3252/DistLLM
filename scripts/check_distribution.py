"""Build a wheel from an sdist and smoke-test its installed console scripts."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def check(sdist):
    with tempfile.TemporaryDirectory(prefix="distllm-package-") as temporary:
        root = Path(temporary)
        source, wheels, site, work = (root / name for name in ("source", "wheels", "site", "work"))
        source.mkdir()
        work.mkdir()
        shutil.unpack_archive(str(sdist), source)
        project, = source.iterdir()
        subprocess.run([sys.executable, "-m", "build", "--wheel", "--no-isolation",
                        "--outdir", str(wheels), str(project)], check=True, cwd=work)
        wheel, = wheels.glob("*.whl")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-index",
                        "--target", str(site), str(wheel)], check=True, cwd=work)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(site)
        probe = subprocess.run([sys.executable, "-c",
            "import json, pathlib, distllm; from importlib.metadata import distribution; "
            "assert pathlib.Path(distllm.__file__).resolve().is_relative_to(pathlib.Path(__import__('sys').argv[1])); "
            "print(json.dumps([e.name for e in distribution('distllm').entry_points if e.group == 'console_scripts']))",
            str(site)], check=True, cwd=work, env=environment, capture_output=True, text=True)
        names = json.loads(probe.stdout)
        assert set(names) == {"distllm-analyze", "distllm-collectives", "distllm-gpu-study",
                              "distllm-reproduce", "distllm-run", "distllm-scaling"}
        suffix = ".exe" if os.name == "nt" else ""
        scripts = next(directory for directory in (site / "Scripts", site / "bin")
                       if (directory / ("distllm-reproduce" + suffix)).exists())
        for name in names:
            executable = scripts / (name + ".exe" if os.name == "nt" else name)
            subprocess.run([str(executable), "--help"], check=True, cwd=work,
                           env=environment, capture_output=True, text=True)
        reproduction = scripts / ("distllm-reproduce.exe" if os.name == "nt" else "distllm-reproduce")
        subprocess.run([str(reproduction), "--output", "reproduction.json"], check=True,
                       cwd=work, env=environment, capture_output=True, text=True)
        result = json.loads((work / "reproduction.json").read_text(encoding="utf-8"))
        assert result["fixed_gradient_relative_l2"] < 1e-10
        print("Distribution smoke check passed: sdist, wheel, imports, six entry points, reproduction")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("sdist", type=Path, nargs="?")
    args = parser.parse_args()
    if args.sdist is None:
        candidates = list(Path("dist").glob("distllm-*.tar.gz"))
        if len(candidates) != 1:
            parser.error("build one source distribution first, or supply its path")
        args.sdist = candidates[0]
    check(args.sdist.resolve())


if __name__ == "__main__":
    main()
