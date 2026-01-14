import scipy.io as sio
print("Loading Exp3.mat")

mat = sio.loadmat("DATA/Exp3.mat", squeeze_me=True, struct_as_record=False)

Exp3 = mat["Exp3"]

print("Loaded Exp3.")
print("Type:", type(Exp3))

fields = [f for f in dir(Exp3) if not f.startswith(" ")]
print("Fields in Exp3:")
for f in fields:
    print(" ", f)

    