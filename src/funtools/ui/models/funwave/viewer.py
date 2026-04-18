from os import PathLike
import param

from pathlib import Path
from os import PathLike

class Viewer(param.Parameterized):


    path =  param.ClassSelector(class_=PathLike)






    def _check_field_data
