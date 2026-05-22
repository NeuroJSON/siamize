% SIAMIZE  Run SIAM v0.3 head/brain MRI segmentation on an in-memory volume.
%
%   labels = siamize(img, affine, models)
%   labels = siamize(img, affine, models, opts)
%
%   img      3D numeric array (uint8/int16/int32/single/double/...).
%            Interpreted as a NIfTI voxel grid with axes [X, Y, Z] in
%            MATLAB's natural column-major layout.
%
%   affine   3x4 or 4x4 double matrix mapping voxel axes to RAS world
%            coordinates. Same convention as jsonlab's loadnifti():
%               nii = loadnifti('input.nii.gz');
%               img = nii.NIFTIData;
%               affine = nii.NIFTIHeader.Affine;
%
%   models   char array (single .onnx path) OR cellstr of .onnx fold
%            paths. Logits are averaged across folds (5-fold ensemble
%            is the upstream-SIAM default).
%
%   opts     optional struct with any of:
%               .device     'auto' (default), 'cpu', 'cuda', 'tensorrt'
%               .threads    int (0 = all available cores, default)
%               .patch      [pz, py, px] (default [256 256 192])
%               .spacing    double mm (default 0.75)
%               .classes    int (default 18, matches SIAM v0.3)
%               .trt_cache  char (default ~/.cache/siamize/trt)
%               .verbose    logical (default false)
%
%   labels   uint8 3D array with the same [X, Y, Z] shape as `img`. Label
%            integers 0..(classes-1). See SIAM's dataset.json for the
%            mapping (0=background, 1=GM, 2=WM, 3=CSF, ... 17=Anomalies).
%
% Example  (round-trip via jsonlab):
%   nii  = loadnifti('input.nii.gz');
%   lab  = siamize(nii.NIFTIData, ...
%                  nii.NIFTIHeader.Affine, ...
%                  {'models/fold_0_fp16.onnx', ...
%                   'models/fold_1_fp16.onnx', ...
%                   'models/fold_2_fp16.onnx', ...
%                   'models/fold_3_fp16.onnx', ...
%                   'models/fold_4_fp16.onnx'});
%   nii_out          = nii;
%   nii_out.NIFTIData = lab;
%   savenifti(nii_out, 'output.nii.gz');
%
% See also: loadnifti, savenifti (https://github.com/NeuroJSON/jsonlab)
%
% siamize project: https://github.com/NeuroJSON/siamize
% SIAM paper:      https://arxiv.org/abs/2605.02737
error('siamize is a MEX function; the .mexa64/.mexw64/.mexmaci64 binary must be on the MATLAB path.');
