"""A PASS review applies to the bundle it judged: Open PR relies on it only
while the record still holds that bundle."""
from __future__ import annotations

from backend.service.routes import account
from backend.spec_worker.bundle import record_fingerprint, review_fingerprint

RECORD = {
    "spec_yaml": "match: {model_id: [acme/Foo]}\n",
    "new_component_yamls": [
        {"path": "backend/arch/components/drafts/b.yaml", "body": "b"},
        {"path": "backend/arch/components/drafts/a.yaml", "body": "a"}],
    "snapshot_json": {"config": {"hidden_size": 8}, "metadata": {"x": 1}},
}


def _pass(fingerprint):
    return [{"verdict": "pass", "bundle_fingerprint": fingerprint}]


def test_the_review_and_the_record_hash_the_same_bundle():
    reviewed = review_fingerprint(RECORD["spec_yaml"],
                                  list(reversed(RECORD["new_component_yamls"])),
                                  {"hidden_size": 8})
    assert reviewed == record_fingerprint(RECORD)
    assert account._latest_review_is_pass(_pass(reviewed), RECORD)


def test_an_edit_after_the_review_needs_a_new_review():
    reviewed = record_fingerprint(RECORD)
    for edited in ({**RECORD, "spec_yaml": RECORD["spec_yaml"] + "# edited\n"},
                   {**RECORD, "new_component_yamls": RECORD["new_component_yamls"][:1]},
                   {**RECORD, "snapshot_json": {"config": {"hidden_size": 16}}}):
        assert not account._latest_review_is_pass(_pass(reviewed), edited)


def test_a_review_from_before_the_binding_does_not_apply():
    assert not account._latest_review_is_pass(_pass(None), RECORD)
    assert not account._latest_review_is_pass(
        [{"verdict": "no_pass", "bundle_fingerprint": record_fingerprint(RECORD)}],
        RECORD)
