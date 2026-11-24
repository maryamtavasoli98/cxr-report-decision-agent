import importlib


def test_package_imports():
    assert importlib.import_module("cxr_report_decision_agent").__version__
