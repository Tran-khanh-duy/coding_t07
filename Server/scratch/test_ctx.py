import contextlib

@contextlib.contextmanager
def my_ctx():
    print("entering")
    yield
    print("exiting (commit!)")

def test():
    with my_ctx():
        print("inside with")
        return "returned value"

print("Result:", test())
