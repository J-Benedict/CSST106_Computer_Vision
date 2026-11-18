"""CLI script to run Labs 1–7 and generate LAB_REPORT.md, .docx, and .html."""

from __future__ import annotations

from utils.report_generator import generate_all_reports


def main() -> None:
    outputs = generate_all_reports()
    print("Generated laboratory reports:")
    for fmt, path in outputs.items():
        print(f"  - [{fmt.upper()}] {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
