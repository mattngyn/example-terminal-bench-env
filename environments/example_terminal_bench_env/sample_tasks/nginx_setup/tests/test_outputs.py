import subprocess
from pathlib import Path


def test_index_page_exists():
    """The static HTML file must exist at the expected path."""
    index = Path("/var/www/html/index.html")
    assert index.exists(), "index.html not found at /var/www/html/index.html"


def test_index_page_content():
    """The page must contain the required text."""
    index = Path("/var/www/html/index.html")
    content = index.read_text()
    assert "Hello Terminal-Bench" in content, (
        f"Expected 'Hello Terminal-Bench' in index.html, got: {content[:200]}"
    )


def test_nginx_listening_on_8080():
    """nginx must be actively serving on port 8080."""
    result = subprocess.run(
        ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "http://localhost:8080"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.stdout.strip() == "200", (
        f"Expected HTTP 200 from localhost:8080, got: {result.stdout.strip()}"
    )


def test_served_content_matches():
    """The content served on port 8080 must contain the required text."""
    result = subprocess.run(
        ["curl", "-s", "http://localhost:8080"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "Hello Terminal-Bench" in result.stdout, (
        f"Expected 'Hello Terminal-Bench' in response, got: {result.stdout[:200]}"
    )
