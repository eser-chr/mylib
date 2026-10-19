from pathlib import Path
import subprocess

# def get_file_path_of(path: Path, expected_word: str) -> Path:
#     """
#     Returns the path of a file based on an expected_word to be in that path.
#     Otherwise it assumes the path is a directory and appends the expected world at the end.
#     """
    
#     if path.name==expected_word:
#         file_path = path
        
#     elif path.name != expected_word and path.is_dir():
#         file_path = path/expected_word
        
#     if not file_path.exists():
#         raise FileNotFoundError()
        
    
#     return file_path
    

class Parse:
    @staticmethod
    def incar(path: Path) -> dict:
        """Parse INCAR file into a dictionary."""
        # file_path = get_file_path_of(path, "INCAR")
      
        with open(path, "r") as f:
            lines = f.readlines()
            
        incar_data = {}
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = line.split("#")[0]
            key_value = line.split("=")
            if len(key_value) == 2:
                key, value = key_value
                incar_data[key.strip()] = value.strip()
        return incar_data

    @staticmethod
    def kpoints(path: Path) -> dict:
        """Parse KPOINTS file into a structured dictionary."""
        # file_path = get_file_path_of(path, "KPOINTS")
     
        with open(path, "r") as f:
            lines = f.readlines()
        
        lines = [line.strip() for line in lines  if line.strip()]

        kpoints_data = {
            "comment": lines[0],
            "num_kpoints": lines[1],
            "generation_mode": lines[2],
            "grid": [int(x) for x in lines[3].split()],
            "shift": [float(x) for x in lines[4].split()] if len(lines) > 4 else None,
        }
        return kpoints_data

    @staticmethod
    def outcar(path: Path) -> dict:
        """Parses the outcar file and returns runtime data."""
        # file_path = get_file_path_of(path, "OUTCAR")
        cpu_time = -1.0
        wall_time = -1.0
        output = subprocess.run(
            ["tail", "-n", "20", path], stdout=subprocess.PIPE, text=True
        )
        output = output.stdout.strip()
        for line in output.split("\n"):
            if "Elapsed time (sec):" in line:
                wall_time = float(line.split(":")[-1].strip())
            elif "Total CPU time used (sec):" in line:
                cpu_time = float(line.split(":")[-1].strip())

            # Stop searching once both values are found
            if cpu_time != -1.0 and wall_time != -1.0:
                break

        return {"cpu_time": cpu_time, "wall_time": wall_time}
    

    @staticmethod
    def potcar(path: Path, verbose=False) -> list:
        """Parses a potcar file and returns the names of the POTCAR files used.
        Args:
        - path: The path to the potcar file or to s folder that contains it
        - verbose (bool): Default(False): If True it prints more info
        """
        # file_path = get_file_path_of(path, "POTCAR")

        with open(path, "r") as f:
            lines = f.readlines()
            
        titles = []
        for line in lines:
            if "TITEL" in line:
                titles.append(line.strip())
                
        if verbose:
            return titles
        else:
            return [title.split()[3] for title in titles]

