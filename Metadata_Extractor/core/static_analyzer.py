import os
import re

# Patterns that mark the start of a test case across the supported languages.
#  - Python / Ruby unittest style:   def test_...
#  - JUnit / TestNG:                  @Test  or  public void test...
#  - RSpec / BDD style:               it "..."  /  it '...'
TEST_CASE_RE = re.compile(
    r'(def\s+test|@Test\b|public\s+void\s+test|\bit\s+[\'"])',
    re.IGNORECASE,
)


def analyze_workspace(repo_path: str) -> dict:
    sloc = 0
    assert_count = 0
    test_lines = 0
    test_cases = 0
    lang_counts = {"java": 0, "python": 0, "ruby": 0}

    ignore_dirs = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build"}

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext == ".py":
                lang_counts["python"] += 1
            elif ext == ".java":
                lang_counts["java"] += 1
            elif ext == ".rb":
                lang_counts["ruby"] += 1
            else:
                continue

            fpath = os.path.join(root, file)
            is_test = "test" in fpath.lower() or "spec" in fpath.lower()
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line_s = line.strip()
                        if line_s and not line_s.startswith("#") and not line_s.startswith("//"):
                            sloc += 1
                            if is_test:
                                test_lines += 1
                                if TEST_CASE_RE.search(line_s):
                                    test_cases += 1
                            if "assert" in line_s:
                                assert_count += 1
            except Exception:
                pass

    kloc = (sloc / 1000) if sloc > 0 else 0
    asserts_per_kloc = (assert_count / kloc) if kloc > 0 else 0.0
    test_lines_per_kloc = (test_lines / kloc) if kloc > 0 else 0.0
    test_cases_per_kloc = (test_cases / kloc) if kloc > 0 else 0.0

    return {
        "gh_sloc": sloc,
        "gh_asserts_cases_per_kloc": round(asserts_per_kloc, 2),
        "gh_test_lines_per_kloc": round(test_lines_per_kloc, 2),
        "gh_test_cases_per_kloc": round(test_cases_per_kloc, 2),
        "gh_lang_java": 1 if lang_counts["java"] > max(lang_counts["python"], lang_counts["ruby"]) else 0,
        "gh_lang_python": 1 if lang_counts["python"] >= max(lang_counts["java"], lang_counts["ruby"]) and lang_counts["python"] > 0 else 0,
        "gh_lang_ruby": 1 if lang_counts["ruby"] > max(lang_counts["java"], lang_counts["python"]) else 0
    }
