from video_parser import video_parser
from image_scissor import image_scissor
from stereo_rectify import stereo_rectify
from depth_to_disp import depth_to_disparity

#rootpath = '/media/eikoloki/TOSHIBA EXT/MICCAI_SCARED/

for i in range(7, 8):
    rootpath = '/media/10TB/EndoVis_depth/dataset_' + str(i)
    print(rootpath)
    video_parser(rootpath)

    #image_scissor(rootpath)
    ##
    #stereo_rectify(rootpath)
    ##
    #depth_to_disparity(rootpath)
