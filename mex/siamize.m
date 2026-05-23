function lab = siamize(varargin)
%SIAMIZE  Run SIAM v0.3 head/brain MRI segmentation on a NIfTI volume.
%
%   labels = siamize(input)
%   labels = siamize(input, models)
%   labels = siamize(input, models, opts)
%   labels = siamize(img,   affine)
%   labels = siamize(img,   affine, models)
%   labels = siamize(img,   affine, models, opts)
%
%   input    can be any of:
%     - a NIfTI file path ('input.nii' or 'input.nii.gz'); siamize calls
%       jsonlab's loadnifti() to read it. If jsonlab is not on the
%       MATLAB/Octave path, siamize auto-adds the bundled copy at
%       <siamize_dir>/mex/jsonlab (`git submodule update --init` in the
%       siamize repo populates that directory).
%     - a jnifti struct with fields .NIFTIData and .NIFTIHeader.Affine,
%       as returned by jsonlab's loadnifti(file).
%     - a readnifti-style struct with fields .img and .hdr (with raw
%       NIfTI header fields srow_x/srow_y/srow_z), as returned by
%       loadnifti(file, 'nii') or nii2jnii(file, 'nii').
%     - a 3D numeric array of voxel intensities. If the next argument
%       is not a 3x4 / 4x4 affine matrix (i.e. it is a models spec,
%       opts struct, or missing), siamize synthesizes an identity-
%       rotation affine that places the world origin at the volume
%       center:
%             A = [1 0 0 -(Nx-1)/2;
%                  0 1 0 -(Ny-1)/2;
%                  0 0 1 -(Nz-1)/2]
%       The translation does not affect the predicted labels (siamize
%       only consumes axes orientation + voxel spacing); it is a
%       cosmetic default for header round-tripping.
%
%   img      3D numeric array (uint8/int16/int32/single/double/...).
%            Interpreted as a NIfTI voxel grid with axes [X, Y, Z] in
%            MATLAB's natural column-major layout.
%
%   affine   3x4 or 4x4 matrix mapping voxel axes to RAS world
%            coordinates (single or double). Same convention as
%            jsonlab's loadnifti():
%               nii = loadnifti('input.nii.gz');
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
%   labels   uint8 3D array with the same [X, Y, Z] shape as the input
%            volume. Label integers 0..17 per SIAM v0.3 (0=background,
%            1=GM, 2=WM, 3=CSF, ..., 17=Anomalies).
%
%   The actual numerical pipeline lives in the companion MEX binary
%   `siamex.mex*` next to this file; `siamize.m` handles input
%   normalization (file/struct/array), jsonlab path injection, model
%   shortcut expansion, weight download via urlwrite + gunzip, and
%   then forwards the call to siamex.
%
% Examples
%   % file path in, round-trip via jsonlab:
%   nii = loadnifti('input.nii.gz');
%   nii.NIFTIData = siamize('input.nii.gz');
%   savenifti(nii, 'labels.nii.gz');
%
%   % full 5-fold ensemble on a jnifti struct:
%   nii = loadnifti('input.nii.gz');
%   lab = siamize(nii, 0:4);
%
%   % pure array, default affine inferred:
%   lab = siamize(my_volume);
%   lab = siamize(my_volume, 0);              % single fold by shortcut
%   lab = siamize(my_volume, 0:4, struct('verbose', true));
%
% See also: siamex, loadnifti, savenifti (https://github.com/NeuroJSON/jsonlab).
%
% siamize project: https://github.com/NeuroJSON/siamize
% SIAM paper:      https://arxiv.org/abs/2605.02737

    siamize_ensure_jsonlab_();
    [img, affine, models, opts] = siamize_parse_inputs_(varargin{:});

    if isempty(models)
        models = {'fold_0_fp16.onnx'};
    elseif isnumeric(models)
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

    resolved = cell(size(models));
    for k = 1:numel(models)
        resolved{k} = siamize_resolve_model_(models{k}, opts);
    end

    here = fileparts(mfilename('fullpath'));
    if ~isempty(here) && exist(fullfile(here, ['siamex.', mexext]), 'file') == 3
        if ~ismember(here, regexp(path, pathsep, 'split'))
            addpath(here);
        end
    end

    lab = siamex(img, affine, resolved, opts);
end


function siamize_ensure_jsonlab_()
%SIAMIZE_ENSURE_JSONLAB_  Add bundled jsonlab to path if loadnifti is missing.
    if exist('loadnifti', 'file') == 2
        return;
    end
    here = fileparts(mfilename('fullpath'));
    jl = fullfile(here, 'jsonlab');
    if exist(jl, 'dir') == 7
        addpath(jl);
    end
end


function [img, affine, models, opts] = siamize_parse_inputs_(varargin)
%SIAMIZE_PARSE_INPUTS_  Normalize flexible inputs into (img, affine, models, opts).
    if nargin < 1
        error('siamize:nargin', ...
              ['usage: siamize(input [, models, opts]) or '...
               'siamize(img, affine [, models, opts])']);
    end

    in = varargin{1};
    if isa(in, 'string') && isscalar(in)
        in = char(in);
    end

    models = [];
    opts = [];

    if ischar(in)
        % NIfTI file path -> loadnifti
        if exist('loadnifti', 'file') ~= 2
            error('siamize:nojsonlab', ...
                  ['loadnifti() not found. Install jsonlab '...
                   '(https://github.com/NeuroJSON/jsonlab) or run '...
                   '`git submodule update --init` so mex/jsonlab is populated.']);
        end
        nii = loadnifti(in);
        img = nii.NIFTIData;
        affine = double(nii.NIFTIHeader.Affine);
        rest = varargin(2:end);
    elseif isstruct(in)
        if isfield(in, 'NIFTIData')
            % jnifti-style struct (loadnifti(file))
            img = in.NIFTIData;
            if isfield(in, 'NIFTIHeader') && isfield(in.NIFTIHeader, 'Affine')
                affine = double(in.NIFTIHeader.Affine);
            else
                affine = siamize_default_affine_(size(img));
            end
        elseif isfield(in, 'img') && isfield(in, 'hdr')
            % readnifti-style struct (loadnifti(file, 'nii'))
            img = in.img;
            affine = siamize_affine_from_niihdr_(in);
        else
            error('siamize:struct', ...
                  ['struct input must be a jnifti struct (.NIFTIData) '...
                   'or a readnifti-style struct (.img + .hdr); '...
                   'got fields: %s'], ...
                  strjoin(fieldnames(in), ','));
        end
        rest = varargin(2:end);
    elseif isnumeric(in)
        img = in;
        if nargin >= 2 && siamize_is_affine_arg_(varargin{2})
            affine = double(varargin{2});
            rest = varargin(3:end);
        else
            affine = siamize_default_affine_(size(img));
            rest = varargin(2:end);
        end
    else
        error('siamize:input', ...
              ['first input must be a file path (char), a jnifti '...
               'struct (.NIFTIData / .NIFTIHeader), or a numeric 3D array']);
    end

    if numel(rest) >= 1
        models = rest{1};
    end
    if numel(rest) >= 2
        opts = rest{2};
    end
    if isempty(opts) || ~isstruct(opts)
        opts = struct();
    end
end


function tf = siamize_is_affine_arg_(x)
%SIAMIZE_IS_AFFINE_ARG_  Numeric 3x4 or 4x4 -> treat as an affine.
    tf = isnumeric(x) && (isequal(size(x), [3, 4]) || isequal(size(x), [4, 4]));
end


function A = siamize_affine_from_niihdr_(nii)
%SIAMIZE_AFFINE_FROM_NIIHDR_  Extract a 3x4 RAS affine from a readnifti-style struct.
%   Stacks hdr.srow_x/srow_y/srow_z (the NIfTI sform), matching
%   jsonlab/niiheader2jnii. If those fields are missing or all-zero
%   (sform not populated by the writer) we fall back to a centered
%   identity; SIAM only consumes rotation + voxel spacing so a missing
%   sform mostly affects the cosmetic origin.
    hdr = nii.hdr;
    has_sform = isfield(hdr, 'srow_x') && isfield(hdr, 'srow_y') ...
                && isfield(hdr, 'srow_z');
    if has_sform
        rows = double([hdr.srow_x(:)'; hdr.srow_y(:)'; hdr.srow_z(:)']);
        if any(rows(:) ~= 0)
            A = rows;
            return;
        end
    end
    A = siamize_default_affine_(size(nii.img));
end


function A = siamize_default_affine_(sz)
%SIAMIZE_DEFAULT_AFFINE_  Identity rotation, translation centers the volume.
%   World origin is placed at the centre voxel ((Nx-1)/2, (Ny-1)/2,
%   (Nz-1)/2) under the standard NIfTI 0-indexed convention.
    if numel(sz) < 3
        error('siamize:shape', 'input volume must be 3D');
    end
    Nx = double(sz(1));
    Ny = double(sz(2));
    Nz = double(sz(3));
    A = single([1, 0, 0, -(Nx - 1) / 2;
                0, 1, 0, -(Ny - 1) / 2;
                0, 0, 1, -(Nz - 1) / 2]);
end


function s = siamize_expand_fold_(s)
%SIAMIZE_EXPAND_FOLD_  '0'..'9' -> 'fold_<N>_fp16.onnx'; pass through otherwise.
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
