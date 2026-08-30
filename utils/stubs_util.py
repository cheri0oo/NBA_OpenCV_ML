import os
import pickle

def save_stubs(stub_path, objects):
    if stub_path is None:
        return

    directory = os.path.dirname(stub_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(stub_path, 'wb') as f:
        pickle.dump(objects, f)

def read_stubs(read_from_stub, stub_path):
    if read_from_stub and stub_path is not None and os.path.exists(stub_path):
        try:
            with open(stub_path, 'rb') as f:
                objects = pickle.load(f)
                return objects
        except Exception:
            # File is empty, corrupted, unreadable, or not a valid pickle
            return None
    return None


