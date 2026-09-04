import pytest

from drummer.documentation import build_reference, project_guides


def test_projection_drift(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "agent-guide.md").write_text("# Guide\n")
    assert project_guides(tmp_path) == ["AGENTS.md", "CLAUDE.md"]
    assert project_guides(tmp_path, check=True) == []
    (tmp_path / "CLAUDE.md").write_text("stale")
    with pytest.raises(ValueError, match="drift"):
        project_guides(tmp_path, check=True)


def test_html_escapes_raw_markup_and_has_semantics(tmp_path):
    root, out = tmp_path / "source", tmp_path / "reference"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "index.md").write_text("# Title\n\n<script>alert(1)</script>\n")
    pages = build_reference(root, out)
    content = open(pages[0]).read()
    assert "<script>" not in content
    assert '<main id="content">' in content
    assert '<html lang="en">' in content


def test_bad_reference_and_checkout_destination_rejected(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "index.md").write_text("# Test\n[bad](missing.md)")
    with pytest.raises(ValueError):
        build_reference(tmp_path, tmp_path / "html")
    with pytest.raises(ValueError, match="Broken"):
        build_reference(tmp_path, tmp_path.parent / (tmp_path.name + "-reference"))


def test_offline_reference_bundles_schema_assets(tmp_path):
    root, out = tmp_path / "source", tmp_path / "html"
    (root / "docs").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "src/packet.json").write_text('{"type":"object"}')
    (root / "docs/index.md").write_text("# Manual\n\n[Schema](../src/packet.json)\n")
    build_reference(root, out)
    assert 'href="resources/src/packet.json"' in (out / "index.html").read_text()
    assert (out / "resources/src/packet.json").read_bytes() == (root / "src/packet.json").read_bytes()
