from pymatgen.ext.matproj import MPRester

MPI_KEY:str = "re1OnovEumCUghRMnWUJafOubEYo7fo1"

def get_material(mpi_id: str, api_key: str):
    mpr = MPRester(api_key)
    entry = mpr.get_entry_by_material_id(mpi_id)
    return entry
