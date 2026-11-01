from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from openplaque import colab_setup


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    fake = SimpleNamespace(version_info=(3, 13, 16), version='3.13.16',
                           executable=sys.executable, modules={})
    monkeypatch.setattr(colab_setup, 'sys', fake)
    monkeypatch.setattr(colab_setup.tempfile, 'tempdir', str(tmp_path))
    monkeypatch.setattr(colab_setup, 'version', lambda name: '2.8.0+cu126' if name == 'torch' else '2.3.4')
    commands = []
    def run(command, check):
        assert check is True
        commands.append(command)
        if '-c' in command:
            constraints = Path(command[command.index('-c') + 1]).read_text()
            assert constraints == 'torch==2.8.0+cu126\n'
    monkeypatch.setattr(colab_setup.subprocess, 'run', run)
    return fake, commands


def test_python313_preserves_cuda_build(runtime, tmp_path):
    fake, commands = runtime
    colab_setup.install_dependencies(tmp_path)
    assert len(commands) == 2
    assert '--prefer-binary' in commands[0]
    assert commands[0][-1] == str(tmp_path / 'requirements-artery-estimates.txt')
    assert '--no-deps' in commands[1]


def test_loaded_package_change_requires_restart(runtime, tmp_path):
    fake, commands = runtime
    fake.modules['numpy'] = SimpleNamespace(__version__='2.0.2')
    with pytest.raises(RuntimeError, match='numpy changed on disk'):
        colab_setup.install_dependencies(tmp_path)


def test_same_revision_can_be_rerun(runtime, tmp_path):
    fake, commands = runtime
    fake.modules['openplaque'] = SimpleNamespace(__file__=str(tmp_path / 'src/openplaque/__init__.py'))
    fake.modules['torch'] = SimpleNamespace(__version__='2.8.0+cu126')
    colab_setup.install_dependencies(tmp_path)


def test_previous_revision_requires_restart(runtime, tmp_path):
    fake, commands = runtime
    fake.modules['openplaque'] = SimpleNamespace(__file__=str(tmp_path / 'other/openplaque/__init__.py'))
    with pytest.raises(RuntimeError, match='updated OpenPlaque'):
        colab_setup.install_dependencies(tmp_path)


def test_install_failure_is_not_ignored(runtime, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise colab_setup.subprocess.CalledProcessError(1, 'pip')
    monkeypatch.setattr(colab_setup.subprocess, 'run', fail)
    with pytest.raises(colab_setup.subprocess.CalledProcessError):
        colab_setup.install_dependencies(tmp_path)
