"""Dependency setup for the audited notebook; load by runpy before package imports."""
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import tempfile


def install_dependencies(repo_dir):
    if sys.version_info[:2] < (3, 10):
        raise RuntimeError(f'Python >=3.10 is required; found {sys.version}')
    # Do not replace the runtime's GPU build with an unrelated PyPI torch wheel.
    torch_version = version('torch')
    with tempfile.TemporaryDirectory(prefix='openplaque-setup-') as temp:
        constraints = Path(temp) / 'runtime-constraints.txt'
        constraints.write_text(f'torch=={torch_version}\n')
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--prefer-binary',
            '--upgrade-strategy', 'only-if-needed',
            '-c', str(constraints), '-r', str(Path(repo_dir) / 'requirements-artery-estimates.txt'),
        ], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-e', str(repo_dir)], check=True)
    for module, distribution in [('numpy', 'numpy'), ('scipy', 'scipy'), ('pandas', 'pandas'), ('torch', 'torch')]:
        loaded = sys.modules.get(module)
        if loaded is not None and str(getattr(loaded, '__version__', '')).split('+')[0] != version(distribution).split('+')[0]:
            raise RuntimeError(f'{module} changed on disk. Restart the Colab session, then run all cells.')
    loaded = sys.modules.get('openplaque')
    expected = Path(repo_dir) / 'src' / 'openplaque' / '__init__.py'
    if loaded is not None and Path(loaded.__file__).resolve() != expected.resolve():
        raise RuntimeError('Restart the session to load the updated OpenPlaque revision.')
    print(f'Python {sys.version.split()[0]}; preserved PyTorch {torch_version}')
