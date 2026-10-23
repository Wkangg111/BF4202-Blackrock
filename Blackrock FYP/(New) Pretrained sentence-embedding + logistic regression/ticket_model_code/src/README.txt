Ticket intake model: files needed to re-run notebooks/escalation_category_model.ipynb

Folder layout expected by the notebook:
  artifacts/model.onnx       all-MiniLM-L6-v2 ONNX weights (90 MB, not included). Download from
                             https://github.com/spring-projects/spring-ai  path:
                             models/spring-ai-transformers/src/main/resources/onnx/all-MiniLM-L6-v2/model.onnx
                             SHA-256 e3dde332c13808c718680e7bf74a574e7e5d06f55bd6e1527e51509dcb8206f3
  artifacts/tokenizer.json   same folder in that repo (included here)
  data/New_Case_details_synthetic_v5.xlsx, data/Case_details_synthetic_v3.1.xlsx
  src/embedder.py            pure-Python WordPiece tokenizer + onnxruntime embedder
  src/build_notebook.py      generates the notebook
  src/run_notebook.py        lightweight executor (or just run the notebook in Jupyter)
  src/build_results_workbook.py  builds artifacts/test_results.xlsx from artifacts/results/*.csv
Requires: python 3.10+, numpy, pandas, scikit-learn, onnxruntime, matplotlib, openpyxl, joblib. Full run ~50 min on 2 CPUs.
