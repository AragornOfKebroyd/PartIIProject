import unittest
import os
import sys

# Ensure Gemini/src is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from export_report import generate_latex_tables, export_pdf_vector_figures, generate_executive_report_summary


class TestStage8(unittest.TestCase):
    def setUp(self):
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))

    def test_latex_table_export(self):
        tex_files = generate_latex_tables(self.output_dir)
        self.assertGreater(len(tex_files), 0)
        for _, path in tex_files.items():
            self.assertTrue(os.path.exists(path))
            with open(path, "r") as f:
                content = f.read()
            self.assertIn("\\begin{table}", content)
            self.assertIn("\\end{table}", content)

    def test_pdf_export(self):
        pdfs = export_pdf_vector_figures(self.output_dir)
        self.assertGreater(len(pdfs), 0)
        for p in pdfs:
            self.assertTrue(os.path.exists(p))

    def test_executive_summary_json(self):
        json_path = generate_executive_report_summary(self.output_dir)
        self.assertTrue(os.path.exists(json_path))
        with open(json_path, "r") as f:
            data = f.read()
        self.assertIn("Compressed Sensing", data)


if __name__ == "__main__":
    unittest.main()
