"""Minimal notebook executor (used because Jupyter/ipykernel cannot be installed in this sandbox).

Runs every code cell of an .ipynb in one shared namespace, in order, and writes standard
nbformat-v4 outputs back into the file: stdout/stderr streams, the value of a cell's last
expression (text/plain + text/html for DataFrames), matplotlib figures as image/png, and errors.
The resulting notebook opens and re-runs unchanged in Jupyter / VS Code.
"""
import ast, base64, contextlib, io, json, sys, time, traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _lines(s):
    return s.splitlines(keepends=True)


def _figures():
    outs = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        outs.append({"output_type": "display_data", "metadata": {},
                     "data": {"image/png": base64.b64encode(buf.getvalue()).decode(),
                              "text/plain": [repr(fig)]}})
    plt.close("all")
    return outs


def _display(obj):
    data = {"text/plain": _lines(repr(obj))}
    if hasattr(obj, "_repr_html_"):
        try:
            html = obj._repr_html_()
            if html:
                data["text/html"] = _lines(html)
        except Exception:
            pass
    return data


def run(path, cwd=None):
    nb = json.load(open(path))
    ns = {"__name__": "__main__"}
    displayed = []

    def display(*objs):  # minimal IPython.display.display replacement
        for o in objs:
            displayed.append({"output_type": "display_data", "metadata": {}, "data": _display(o)})
    ns["display"] = display

    count = 0
    t_all = time.time()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        count += 1
        src = "".join(cell["source"])
        outputs, out, err = [], io.StringIO(), io.StringIO()
        displayed.clear()
        t0 = time.time()
        result, failed = None, None
        try:
            tree = ast.parse(src)
            last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            body = ast.Module(body=tree.body[:-1] if last else tree.body, type_ignores=[])
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(compile(body, f"<cell {count}>", "exec"), ns)
                if last is not None:
                    result = eval(compile(ast.Expression(last.value), f"<cell {count}>", "eval"), ns)
        except Exception as e:
            failed = e
        if out.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout", "text": _lines(out.getvalue())})
        if err.getvalue():
            outputs.append({"output_type": "stream", "name": "stderr", "text": _lines(err.getvalue())})
        outputs.extend(displayed)
        outputs.extend(_figures())
        if failed is not None:
            tb = traceback.format_exception(type(failed), failed, failed.__traceback__)
            outputs.append({"output_type": "error", "ename": type(failed).__name__,
                            "evalue": str(failed), "traceback": tb})
        elif result is not None:
            outputs.append({"output_type": "execute_result", "execution_count": count,
                            "metadata": {}, "data": _display(result)})
        cell["outputs"], cell["execution_count"] = outputs, count
        print(f"cell {count:2d} done in {time.time() - t0:6.1f}s" + (f"  ERROR: {failed!r}" if failed else ""),
              flush=True)
        json.dump(nb, open(path, "w"), indent=1)
        if failed is not None:
            print("".join(tb), file=sys.stderr)
            return 1
    print(f"all cells done in {time.time() - t_all:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
