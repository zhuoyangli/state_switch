#! /bin/bash
# use freesurfer mri_surf2surf to smooth functional images preprocessed with fmriprep
# to be run on the lab server 
# hongmi lee 5/30/22

EXPDIR=/storage/hongmi/ThinkAloud/derivatives

subjects=( 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 )
nsubject=${#subjects[@]}
smoothfwhm=4

for (( ii=0; ii<${nsubject}; ii++)); do

    SN=sub-${subjects[ii]}
    
    # surface smoothing
    mri_surf2surf --s fsaverage6 --hemi lh --sval ${EXPDIR}/${SN}/func/${SN}_task-thinkaloud_hemi-L_space-fsaverage6_bold.func.gii --fwhm ${smoothfwhm} --tval ${EXPDIR}/${SN}/func/${SN}_task-thinkaloud_hemi-L_space-fsaverage6_desc-sm${smoothfwhm}_bold.func.gii
    mri_surf2surf --s fsaverage6 --hemi rh --sval ${EXPDIR}/${SN}/func/${SN}_task-thinkaloud_hemi-R_space-fsaverage6_bold.func.gii --fwhm ${smoothfwhm} --tval ${EXPDIR}/${SN}/func/${SN}_task-thinkaloud_hemi-R_space-fsaverage6_desc-sm${smoothfwhm}_bold.func.gii
    
#    # duplicate json file
#    cp ${EXPDIR}/${SN}/func/${SN}_task-thinkaloud_hemi-L_space-fsaverage6_bold.json ${EXPDIR}/${SN}/func/${SN}_task-thinkaloud_hemi-L_space-fsaverage6_desc-sm${smoothfwhm}_bold.json
#    cp ${EXPDIR}/${SN}/func/${SN}_task-thinkaloud_hemi-R_space-fsaverage6_bold.json ${EXPDIR}/${SN}/func/${SN}_task-thinkaloud_hemi-R_space-fsaverage6_desc-sm${smoothfwhm}_bold.json    
done