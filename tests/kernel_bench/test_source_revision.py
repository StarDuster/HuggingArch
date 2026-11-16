from types import SimpleNamespace
import subprocess

from kernel_bench.env import _probe_version


def test_unreleased_source_records_commit_and_tracked_edits(tmp_path, monkeypatch):
    root = tmp_path / 'checkout'
    root.mkdir()
    module = root / 'pkg.py'
    module.write_text('__version__ = "0.0.0"\n')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    git('init', '-q')
    git('add', 'pkg.py')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.com',
        'commit', '-qm', 'test source')
    sha = git('rev-parse', 'HEAD')
    parent = SimpleNamespace(__version__='0.0.0', __file__=str(module))
    child = SimpleNamespace(__file__=str(module))
    monkeypatch.setattr('kernel_bench.env.importlib.import_module',
                        lambda name: parent if name == 'pkg' else child)
    assert _probe_version('pkg.child') == 'git:' + sha
    module.write_text('__version__ = "0.0.0"\n# modified implementation\n')
    assert _probe_version('pkg.child').startswith('git:' + sha + '+dirty.')
    parent.__version__ = '1.2.3'
    assert _probe_version('pkg.child') == '1.2.3'
