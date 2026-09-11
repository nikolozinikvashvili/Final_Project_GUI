import acquisition 

project = acquisition.project

from preprocessing import preprocess

preprocess(project)

from reconstruction import reconstruct

reconstruct(project)

from segmentation import segment

segment(project)