import acquisition 

project = acquisition.project

from preprocessing import preprocess

preprocess(project)

from reconstruction import reconstruct

reconstruct(project)
