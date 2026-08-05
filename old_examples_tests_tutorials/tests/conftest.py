def pytest_assertion_pass(item, lineno, orig, expl):
    print("asserting that {}, {}, {}, {}".format(item, lineno, orig, expl))
