import tarfile

from scripts.publish_sol_pool import build


def test_sol_artifact_is_digestable_and_keeps_bundle_files(tmp_path):
    root = tmp_path / "sol"
    bundle = root / "ab" / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    (bundle / "records-00000.jsonl").write_text("{}\n", encoding="utf-8")
    archive = tmp_path / "pool.tar.gz"

    assert build(root, archive) == 1
    with tarfile.open(archive, "r:gz") as tar:
        assert tar.getnames() == [
            "sol", "sol/ab", "sol/ab/bundle",
            "sol/ab/bundle/manifest.json",
            "sol/ab/bundle/records-00000.jsonl",
        ]
