from ase.io import read, write
from pathlib import Path



if __name__=="__main__":
    print("Enter")
    base = Path("./TRAJ")

    all_structs = []

    for file in base.glob("*.extxyz"):
        atoms = read(file, index=":")
        all_structs.extend(atoms)
    print("Done collecting!")
    write(base/"all.extxyz", all_structs)
    print("Done")
