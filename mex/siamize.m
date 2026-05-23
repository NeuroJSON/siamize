function lab = siamize(img, affine, models, opts)
%SIAMIZE  Run SIAM v0.3 head/brain MRI segmentation on an in-memory volume.
%
%   labels = siamize(img, affine)
%   labels = siamize(img, affine, models)
%   labels = siamize(img, affine, models, opts)
%
%   img      3D numeric array (uint8/int16/int32/single/double/...).
%            Interpreted as a NIfTI voxel grid with axes [X, Y, Z] in
%            MATLAB's natural column-major layout.
%
%   affine   3x4 or 4x4 matrix mapping voxel axes to RAS world
%            coordinates (single or double). Same convention as
%            jsonlab's loadnifti():
%               nii = loadnifti('input.nii.gz');
%               img = nii.NIFTIData;
%               affine = nii.NIFTIHeader.Affine;
%
%   models   [] | numeric | char | cellstr -- one .onnx fold per entry;
%            logits are averaged across the list. Defaults to
%            {'fold_0_fp16.onnx'} (single fold) when omitted or empty.
%            Each entry can be a full path, a bare basename, or a
%            single-digit shortcut (numeric 0..9 or char '0'..'9')
%            which expands to 'fold_<N>_fp16.onnx'. Examples:
%                siamize(img, A, 0)          % single-fold fold_0
%                siamize(img, A, 0:4)        % full 5-fold ensemble
%                siamize(img, A, '0,2,4')    % comma string shortcut
%                siamize(img, A, {'0','2'})  % cellstr shortcut
%            Bare basenames / shortcuts are looked up in the shared cache
%                $SIAMIZE_CACHE_DIR
%                  (default $HOME/.cache/siamize/models on POSIX,
%                   %LOCALAPPDATA%/siamize/models on Windows)
%            and auto-downloaded from
%                $SIAMIZE_WEIGHTS_BASE_URL
%                  (default https://neurojson.org/siamize/weights/siam_v03)
%            via MATLAB/Octave's urlwrite + gunzip. The cache is shared
%            with the siamize CLI binary so a single download serves both.
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
%   labels   uint8 3D array with the same [X, Y, Z] shape as `img`.
%            Label integers 0..17 per SIAM v0.3 (0=background, 1=GM,
%            2=WM, 3=CSF, ..., 17=Anomalies).
%
%   The actual numerical pipeline lives in the companion MEX binary
%   `siamex.mex*` next to this file; `siamize.m` only resolves model
%   files and forwards the call.
%
% Example (with jsonlab):
%   nii  = loadnifti('input.nii.gz');
%   lab  = siamize(nii.NIFTIData, nii.NIFTIHeader.Affine);
%   out  = nii;  out.NIFTIData = lab;
%   savenifti(out, 'output.nii.gz');
%
% See also: siamex, loadnifti, savenifti (https://github.com/NeuroJSON/jsonlab).
%
% siamize project: https://github.com/NeuroJSON/siamize
% SIAM paper:      https://arxiv.org/abs/2605.02737

    if nargin < 2
        error('siamize:nargin', ...
              'usage: labels = siamize(img, affine [, models, opts])');
    end
    if nargin < 3 || isempty(models)
        models = {'fold_0_fp16.onnx'};
    elseif isnumeric(models)
        % siamize(img, A, 0) or siamize(img, A, [0 1 2 3 4])
        nums = models(:);
        models = cell(numel(nums), 1);
        for k = 1:numel(nums)
            n = double(nums(k));
            if n < 0 || n > 9 || n ~= round(n)
                error('siamize:models', ...
                      'numeric model index must be an integer 0..9, got %g', n);
            end
            models{k} = sprintf('fold_%d_fp16.onnx', n);
        end
    elseif ischar(models)
        % '0', '0,1,2,3,4', or a literal path; split on comma either way.
        parts = strsplit(strtrim(models), ',');
        models = cell(numel(parts), 1);
        for k = 1:numel(parts)
            models{k} = siamize_expand_fold_(strtrim(parts{k}));
        end
    elseif iscell(models)
        for k = 1:numel(models)
            if ~ischar(models{k})
                error('siamize:models', ...
                      'cell entry %d must be a char string', k);
            end
            models{k} = siamize_expand_fold_(strtrim(models{k}));
        end
    else
        error('siamize:models', ...
              'models must be [] | numeric | char | cellstr');
    end
    if ~iscellstr(models)
        error('siamize:models', 'models must resolve to a cellstr');
    end
    if nargin < 4
        opts = struct();
    end

    % Resolve / auto-download each model into the shared cache.
    resolved = cell(size(models));
    for k = 1:numel(models)
        resolved{k} = siamize_resolve_model_(models{k}, opts);
    end

    % Ensure siamex.mex* is on the path (the .mex* binary sits next to
    % this .m file when installed normally).
    here = fileparts(mfilename('fullpath'));
    if ~isempty(here) && exist(fullfile(here, ['siamex.', mexext]), 'file') == 3
        if ~ismember(here, regexp(path, pathsep, 'split'))
            addpath(here);
        end
    end

    lab = siamex(img, affine, resolved, opts);
end


function s = siamize_expand_fold_(s)
%SIAMIZE_EXPAND_FOLD_ '0'..'9' -> 'fold_<N>_fp16.onnx'; pass through otherwise.
    if numel(s) == 1 && s >= '0' && s <= '9'
        s = sprintf('fold_%s_fp16.onnx', s);
    end
end


function p = siamize_resolve_model_(spec, opts)
%SIAMIZE_RESOLVE_MODEL_ Locate or download an .onnx fold file.
    if exist(spec, 'file') == 2
        p = spec;
        return;
    end

    cachedir = getenv('SIAMIZE_CACHE_DIR');
    if isempty(cachedir)
        if ispc
            base = getenv('LOCALAPPDATA');
            if isempty(base)
                base = getenv('USERPROFILE');
            end
            cachedir = fullfile(base, 'siamize', 'models');
        else
            cachedir = fullfile(getenv('HOME'), '.cache', 'siamize', 'models');
        end
    end

    [~, name, ext] = fileparts(spec);
    basename = [name, ext];
    p = fullfile(cachedir, basename);
    if exist(p, 'file') == 2
        return;
    end

    if ~exist(cachedir, 'dir')
        mkdir(cachedir);
    end

    urlbase = getenv('SIAMIZE_WEIGHTS_BASE_URL');
    if isempty(urlbase)
        urlbase = 'https://neurojson.org/siamize/weights/siam_v03';
    end

    verbose = false;
    if isstruct(opts) && isfield(opts, 'verbose') && ~isempty(opts.verbose)
        verbose = logical(opts.verbose);
    end

    url_gz = [urlbase, '/', basename, '.gz'];
    tmp_gz = [p, '.gz'];
    fetched_gz = false;
    try
        if verbose
            fprintf('siamize: fetching %s\n', url_gz);
        end
        urlwrite(url_gz, tmp_gz);
        % Verify gzip magic bytes (0x1F 0x8B) so we don't pass an HTML
        % error page through gunzip.
        fid = fopen(tmp_gz, 'rb');
        magic = fread(fid, 2, 'uint8');
        fclose(fid);
        if numel(magic) ~= 2 || magic(1) ~= 31 || magic(2) ~= 139
            error('siamize:notgzip', ...
                  'response from %s is not a gzip file (got %d bytes starting %02x %02x)', ...
                  url_gz, numel(magic), magic(1), magic(2));
        end
        gunzip(tmp_gz, cachedir);
        fetched_gz = true;
    catch err
        if verbose
            fprintf('siamize: .gz path failed (%s); trying raw .onnx\n', err.message);
        end
    end
    if exist(tmp_gz, 'file')
        delete(tmp_gz);
    end

    if fetched_gz && exist(p, 'file') == 2
        return;
    end

    % Fall back to the raw uncompressed URL.
    url_raw = [urlbase, '/', basename];
    if verbose
        fprintf('siamize: fetching %s\n', url_raw);
    end
    urlwrite(url_raw, p);
    if exist(p, 'file') ~= 2
        error('siamize:fetch', ...
              'could not locate or download %s\n  tried %s\n        %s', ...
              basename, url_gz, url_raw);
    end
end
