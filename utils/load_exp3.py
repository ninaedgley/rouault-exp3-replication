import scipy.io as sio
print("Loading Exp3.mat")

def load_exp3(mat_path = "DATA/Exp3.mat"):
    """
    Load Rouault Exp3 MATLAB struct and return it
    """
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    return mat["Exp3"]

if __name__ == "__main__":
    Exp3 = load_exp3()
    print("Loaded Exp3.")
    print("Type:", type(Exp3))

    fields = [f for f in dir(Exp3) if not f.startswith("_")]
    print("Fields in Exp3:")
    for f in fields:
        print(" ", f)
