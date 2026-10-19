import numpy as np
from pathlib import Path
import ase.constraints
from ase import Atoms
from ase.io import read, write
from .parsers import Parse
from typing import cast


def get_fixed_atom_array(atoms:Atoms)->np.ndarray:
    """ 
    Returns a bool array of length equal to the number of atoms, where True means 
    the atom is marked as fixed.
    """
    tmp = np.zeros(len(atoms), dtype = bool)
    if len(atoms.constraints)!=0:
        if type(atoms.constraints[0])==ase.constraints.FixAtoms:
            fixed_atoms = atoms.constraints[0].index
            tmp[fixed_atoms] = True
    return tmp


def vasp_to_extxyz(folder_path:Path, output_path:Path, title:str|None)->None:
    """ 
    Compresses a complete VASP simulation folder to a single extxyz file.
    The extxyz file contains two structures corresponding to initial and final condition
    (even if it is a single point calculation).
    
    Input:
    folder_path: path to the vasp folder.
    output_path: path to the output file
    title (Optional): a string that overwrites the incar SYSTEM command 
                    and serves as a place for potential comments.
    
    """
    poscar = read(folder_path/"POSCAR")
    contcar = read(folder_path/"CONTCAR")
    
    if isinstance(poscar, list):
        raise ValueError("POSCAR contained multiple frames")
    if isinstance(contcar, list):
        raise ValueError("POSCAR contained multiple frames")

    
    
    poscar.arrays["fix_atoms"] = get_fixed_atom_array(poscar)
    contcar.arrays["fix_atoms"] = get_fixed_atom_array(contcar)
    
    
    tmp = Parse.incar(folder_path)
    if title:
        tmp["SYSTEM"] = title
    
    poscar.info["incar"] =  tmp
    
    poscar.info["kpoints"] =  Parse.kpoints(folder_path)
    poscar.info["potcar"] =  Parse.potcar(folder_path)
    contcar.info["wall_time"] = Parse.outcar(folder_path)["wall_time"]
    
    
    write(output_path, [poscar, contcar])