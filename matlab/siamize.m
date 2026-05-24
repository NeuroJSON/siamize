function [lab, tpm] = siamize(varargin)
% SIAMIZE  Run SIAM v0.3 head/brain MRI segmentation on a NIfTI volume.
%
%   labels = siamize(input)
%   labels = siamize(input, outputfile)
%   labels = siamize(input, outputfile, models)
%   labels = siamize(input, outputfile, models, opts)
%   labels = siamize(input, models)
%   labels = siamize(input, models, opts)
%   labels = siamize(img,   affine)
%   labels = siamize(img,   affine, outputfile)
%   labels = siamize(img,   affine, outputfile, models, opts)
%   labels = siamize(img,   affine, models)
%   labels = siamize(img,   affine, models, opts)
%
%   input    can be any of:
%     - a file path readable by jsonlab's loadjd():
%         .nii, .nii.gz   - NIfTI-1/2 (read via loadnifti)
%         .jnii           - text JNIfTI (read via loadjson)
%         .bnii           - binary JNIfTI (read via loadbj)
%       jsonlab must be on the MATLAB/Octave path; if not, siamize
%       auto-adds the bundled copy at <siamize_dir>/matlab/jsonlab
%       (`git submodule update --init` populates it).
%     - a jnifti struct (.NIFTIData + .NIFTIHeader.Affine), as returned
%       by loadnifti(file) / loadjnifti(file).
%     - a readnifti-style struct (.img + .hdr with raw srow_x/y/z), as
%       returned by loadnifti(file, 'nii') or nii2jnii(file, 'nii').
%     - a 3D numeric array of voxel intensities. If the next argument
%       isn't a 3x4/4x4 affine, an identity-rotation affine with the
%       world origin at the volume center is synthesized:
%             A = [1 0 0 -(Nx-1)/2;
%                  0 1 0 -(Ny-1)/2;
%                  0 0 1 -(Nz-1)/2]
%       Translation doesn't affect SIAM predictions (the pipeline only
%       consumes axes orientation + voxel spacing); the default is
%       cosmetic for header round-tripping.
%
%   outputfile  optional char. When present, the predicted label volume
%               is written to disk. Extension picks the writer:
%                 .nii, .nii.gz - jnii2nii(jnii, outputfile)
%                 .jnii, .bnii  - savejnifti(jnii, outputfile)
%               Examples:
%                 siamize('in.nii.gz', 'labels.nii.gz')
%                 siamize('in.jnii',   'labels.bnii', 0:4)
%               When the input is a struct or file, its full NIFTIHeader
%               is preserved in the output (with NIFTIData swapped to
%               the labels). When the input is a bare array, a minimal
%               header is synthesized via jnifticreate and the affine
%               (custom or default-centered) is written into it.
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
%            {'fold_0_fp16.onnx'} when omitted or empty. Each entry can
%            be a full path, a bare basename, or a single-digit
%            shortcut (numeric 0..9 or char '0'..'9') which expands to
%            'fold_<N>_fp16.onnx'. Examples:
%                siamize(in, out, 0)        % single-fold fold_0
%                siamize(in, out, 0:4)      % full 5-fold ensemble
%                siamize(in, out, '0,2,4')  % comma string shortcut
%                siamize(in, out, {'0','2'})
%            Bare basenames / shortcuts are looked up in the shared
%            cache $SIAMIZE_CACHE_DIR (default $HOME/.cache/siamize/
%            models on POSIX, %LOCALAPPDATA%/siamize/models on Windows)
%            and auto-downloaded from $SIAMIZE_WEIGHTS_BASE_URL (default
%            https://neurojson.org/io/stat.cgi?action=get&db=siam_v03&doc=dynshape&size=95360591&file=)
%            via MATLAB/Octave's urlwrite + gunzip. The URL prefix is
%            concatenated with `<basename>.gz` directly (no `/` between).
%
%   opts     optional. Either a struct, a series of 'name', value
%            pairs, or any mix of structs and 'name', value pairs that
%            jsonlab's varargin2struct can merge. All keys are case-
%            insensitive (varargin2struct lowercases). Recognized:
%
%               'device'              'auto' (default), 'cpu', 'cuda', 'tensorrt'
%               'threads'             int (0 = all available cores, default)
%               'patch'               [pz, py, px] (default [256 256 192])
%               'spacing'             double mm (default 0.75)
%               'classes'             int (default 18, matches SIAM v0.3)
%               'trt_cache'           char (default ~/.cache/siamize/trt)
%               'verbose'             logical (default false)
%
%            CUDA EP tuning (only used when device involves CUDA, see
%            the CLI flags --cudnn-* for the same knobs):
%
%               'cudnn_max_workspace' 0 or 1 (default 1 = ORT default).
%                                      Use 0 on tight-VRAM GPUs.
%               'arena_extend'        'power' (default) or 'same'.
%               'cudnn_algo'          'default' (default) | 'heuristic'
%                                      | 'exhaustive'.
%               'gpu_mem_limit'       bytes (double, e.g. 6*1024^3 for 6GB).
%                                      Default 0 = no cap.
%
%            TPM output (4D float32 tissue probability map, softmax
%            over the 18 fold-averaged class logits):
%
%               'tpm_out'             char output path (.nii(.gz) /
%                                      .jnii / .bnii). When set, the
%                                      TPM is saved to disk in
%                                      addition to (or instead of)
%                                      returning it.
%               'tpm_temperature'     double (default 1.0). T>1 softens
%                                      the softmax (more graded
%                                      boundaries, better calibration).
%
%   labels   uint8 3D array with the same [X, Y, Z] shape as the input
%            volume. Label integers 0..17 per SIAM v0.3 (0=background,
%            1=GM, 2=WM, 3=CSF, ..., 17=Anomalies). Always returned;
%            also written to outputfile if one was provided.
%
%   tpm      (optional second output) 4D single (float32) of shape
%            [X, Y, Z, 18] holding the softmax probabilities per voxel
%            and class. Computed and returned only when the caller
%            requests two outputs OR when opts.tpm_out is set. Sums
%            to 1 per voxel; argmax(tpm, 4)-1 reproduces `labels`.
%
% Examples
%   % one-shot file -> file:
%   siamize('input.nii.gz', 'labels.nii.gz');
%
%   % cross-format: read .nii.gz, write binary JNIfTI:
%   siamize('input.nii.gz', 'labels.bnii');
%
%   % struct in, file out, full ensemble, verbose, CUDA EP:
%   nii = loadnifti('input.nii.gz');
%   siamize(nii, 'labels.jnii', 0:4, 'device', 'cuda', 'verbose', true);
%
%   % tight-VRAM GPU: shrink cuDNN workspace, dynamic-shape patch:
%   siamize('in.nii.gz', 'lab.nii.gz', 0:4, ...
%           'device', 'cuda', 'cudnn_max_workspace', 0, ...
%           'arena_extend', 'same', 'patch', [192 192 128]);
%
%   % save 4D TPM to disk alongside the labelmap:
%   siamize('in.nii.gz', 'lab.nii.gz', 0:4, 'tpm_out', 'tpm.nii.gz');
%
%   % return TPM in-memory (second nargout); no file written:
%   [lab, tpm] = siamize('in.nii.gz', 0:4, 'tpm_temperature', 1.5);
%
%   % opts struct still works (BC); can mix with name/value overrides:
%   defs = struct('device', 'cuda', 'verbose', true);
%   siamize('in.nii.gz', 'lab.nii.gz', 0:4, defs, 'tpm_out', 'tpm.bnii');
%
%   % pure array, default affine inferred:
%   lab = siamize(my_volume);
%   lab = siamize(my_volume, 0);
%
% See also: siamex, loadjd, savejd, loadnifti, jnii2nii, savejnifti
% (https://github.com/NeuroJSON/jsonlab).
%
% siamize project: https://github.com/NeuroJSON/siamize
% SIAM paper:      https://arxiv.org/abs/2605.02737

siamize_ensure_jsonlab_();
[img, affine, src, outputfile, models, opts] = siamize_parse_inputs_(varargin{:});

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

% Decide whether to request the optional 4D TPM. Compute it iff the
% caller asked for it via the second nargout or via opts.tpm_out (in
% which case we save it to disk after).
tpm_out = jsonopt('tpm_out', '', opts);
want_tpm = (nargout >= 2) || ~isempty(tpm_out);

if want_tpm
    [lab, tpm] = siamex(img, affine, resolved, opts);
else
    lab = siamex(img, affine, resolved, opts);
end

if ~isempty(outputfile)
    siamize_write_output_(lab, affine, src, outputfile);
end

if ~isempty(tpm_out)
    siamize_write_tpm_(tpm, affine, src, tpm_out);
end
end

function siamize_ensure_jsonlab_()
% SIAMIZE_ENSURE_JSONLAB_  Add bundled jsonlab to path if loadjd is missing.
if exist('loadjd', 'file') == 2
    return
end
here = fileparts(mfilename('fullpath'));
jl = fullfile(here, 'jsonlab');
if exist(jl, 'dir') == 7
    addpath(jl);
end
end

function [img, affine, src, outputfile, models, opts] = siamize_parse_inputs_(varargin)
% SIAMIZE_PARSE_INPUTS_  Normalize flexible inputs into (img, affine, src, outputfile, models, opts).
%   src is the jnifti source struct used as a header template when
%   writing to outputfile; struct() if not available (bare-array input).
if nargin < 1
    error('siamize:nargin', ...
          ['usage: siamize(input [, outputfile, models, opts]) or '...
           'siamize(img, affine [, outputfile, models, opts])']);
end

in = varargin{1};
if isa(in, 'string') && isscalar(in)
    in = char(in);
end

outputfile = '';
models = [];
opts = [];
src = struct();

if ischar(in)
    if exist('loadjd', 'file') ~= 2
        error('siamize:nojsonlab', ...
              ['loadjd() not found. Install jsonlab '...
               '(https://github.com/NeuroJSON/jsonlab) or run '...
               '`git submodule update --init` so matlab/jsonlab is populated.']);
    end
    nii = loadjd(in);
    if ~isstruct(nii) || ~isfield(nii, 'NIFTIData')
        error('siamize:badload', ...
              'loadjd(%s) did not return a jnifti struct (got %s)', ...
              in, class(nii));
    end
    img = nii.NIFTIData;
    if isfield(nii, 'NIFTIHeader') && isfield(nii.NIFTIHeader, 'Affine')
        affine = double(nii.NIFTIHeader.Affine);
    else
        affine = siamize_default_affine_(size(img));
    end
    src = nii;
    rest = varargin(2:end);
elseif isstruct(in)
    if isfield(in, 'NIFTIData')
        img = in.NIFTIData;
        if isfield(in, 'NIFTIHeader') && isfield(in.NIFTIHeader, 'Affine')
            affine = double(in.NIFTIHeader.Affine);
        else
            affine = siamize_default_affine_(size(img));
        end
        src = in;
    elseif isfield(in, 'img') && isfield(in, 'hdr')
        img = in.img;
        affine = siamize_affine_from_niihdr_(in);
        try
            src = niiheader2jnii(in);
            src.NIFTIData = img;
        catch
            src = struct();
        end
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
           'struct (.NIFTIData / .NIFTIHeader), a readnifti-style '...
           'struct (.img / .hdr), or a numeric 3D array']);
end

% Optional output filename: char matching a known image extension.
if numel(rest) >= 1 && siamize_is_image_filename_(rest{1})
    outputfile = rest{1};
    rest = rest(2:end);
end

if numel(rest) >= 1
    models = rest{1};
end

% Everything after `models` is the options bag. Built with jsonlab's
% `varargin2struct` so callers can mix:
%     a single struct:                       siamize(in, out, 0, struct('device','cuda'))
%     'param', value pairs:                  siamize(in, out, 0, 'device','cuda','verbose',1)
%     a struct + name/value overrides:       siamize(in, out, 0, defs, 'verbose', 1)
%     nothing (empty opts):                  siamize(in, out, 0)
opts_args = rest(2:end);
opts = varargin2struct(opts_args{:});
end

function siamize_write_output_(lab, affine, src, outputfile)
% SIAMIZE_WRITE_OUTPUT_  Save the label volume to a .nii(.gz) / .jnii / .bnii file.
%   Preserves the source jnifti header when available; otherwise builds
%   a minimal header via jnifticreate and writes the affine into it.
if isstruct(src) && isfield(src, 'NIFTIHeader')
    jnii_out = src;
    jnii_out.NIFTIData = lab;
    if isfield(jnii_out.NIFTIHeader, 'Affine')
        jnii_out.NIFTIHeader.Affine = affine;
    end
else
    jnii_out = jnifticreate(lab);
    jnii_out.NIFTIHeader.Affine = affine;
end

if ~isempty(regexpi(outputfile, '\.nii(\.gz)?$', 'once'))
    jnii2nii(jnii_out, outputfile);
elseif ~isempty(regexpi(outputfile, '\.(jnii|bnii)$', 'once'))
    savejnifti(jnii_out, outputfile);
else
    error('siamize:outext', ...
          ['unsupported output extension: %s (expected '...
           '.nii / .nii.gz / .jnii / .bnii)'], outputfile);
end
end


function siamize_write_tpm_(tpm, affine, src, outputfile)
% SIAMIZE_WRITE_TPM_  Save a 4D float32 TPM volume to disk.
%   Same writer dispatch as for labels (.nii(.gz) via jnii2nii;
%   .jnii / .bnii via savejnifti). Header inherited from the source
%   jnifti struct when available; otherwise synthesized via
%   jnifticreate so the affine round-trips.
if isstruct(src) && isfield(src, 'NIFTIHeader')
    jnii_out = src;
    jnii_out.NIFTIData = tpm;
    if isfield(jnii_out.NIFTIHeader, 'Affine')
        jnii_out.NIFTIHeader.Affine = affine;
    end
else
    jnii_out = jnifticreate(tpm);
    jnii_out.NIFTIHeader.Affine = affine;
end

if ~isempty(regexpi(outputfile, '\.nii(\.gz)?$', 'once'))
    jnii2nii(jnii_out, outputfile);
elseif ~isempty(regexpi(outputfile, '\.(jnii|bnii)$', 'once'))
    savejnifti(jnii_out, outputfile);
else
    error('siamize:tpmext', ...
          ['unsupported tpm_out extension: %s (expected '...
           '.nii / .nii.gz / .jnii / .bnii)'], outputfile);
end
end

function tf = siamize_is_image_filename_(s)
% SIAMIZE_IS_IMAGE_FILENAME_  True if s names a NIfTI / JNIfTI file by extension.
tf = ischar(s) && ~isempty(regexpi(s, '\.(nii(\.gz)?|jnii|bnii)$', 'once'));
end

function tf = siamize_is_affine_arg_(x)
% SIAMIZE_IS_AFFINE_ARG_  Numeric 3x4 or 4x4 -> treat as an affine.
tf = isnumeric(x) && (isequal(size(x), [3, 4]) || isequal(size(x), [4, 4]));
end

function A = siamize_affine_from_niihdr_(nii)
% SIAMIZE_AFFINE_FROM_NIIHDR_  Extract a 3x4 RAS affine from a readnifti-style struct.
hdr = nii.hdr;
has_sform = isfield(hdr, 'srow_x') && isfield(hdr, 'srow_y') && isfield(hdr, 'srow_z');
if has_sform
    rows = double([hdr.srow_x(:)'; hdr.srow_y(:)'; hdr.srow_z(:)']);
    if any(rows(:) ~= 0)
        A = rows;
        return
    end
end
A = siamize_default_affine_(size(nii.img));
end

function A = siamize_default_affine_(sz)
% SIAMIZE_DEFAULT_AFFINE_  Identity rotation, translation centers the volume.
if numel(sz) < 3
    error('siamize:shape', 'input volume must be 3D');
end
Nx = double(sz(1));
Ny = double(sz(2));
Nz = double(sz(3));
A = single([1, 0, 0, -(Nx - 1) / 2
            0, 1, 0, -(Ny - 1) / 2
            0, 0, 1, -(Nz - 1) / 2]);
end

function s = siamize_expand_fold_(s)
% SIAMIZE_EXPAND_FOLD_  '0'..'9' -> 'fold_<N>_fp16.onnx'; pass through otherwise.
if numel(s) == 1 && s >= '0' && s <= '9'
    s = sprintf('fold_%s_fp16.onnx', s);
end
end

function p = siamize_resolve_model_(spec, opts)
% SIAMIZE_RESOLVE_MODEL_ Locate or download an .onnx fold file.
if exist(spec, 'file') == 2
    p = spec;
    return
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
    return
end

if ~exist(cachedir, 'dir')
    mkdir(cachedir);
end

% NeuroJSON URL prefix; the filename is appended directly (no `/`
% separator) since the prefix ends with `&file=`. The size= query
% parameter is informational and not validated by the server, so a
% single constant covers every fold.
urlbase = getenv('SIAMIZE_WEIGHTS_BASE_URL');
if isempty(urlbase)
    urlbase = ['https://neurojson.org/io/stat.cgi?action=get&db=siam_v03', ...
               '&doc=dynshape&size=95360591&file='];
end

verbose = logical(jsonopt('verbose', false, opts));

url_gz = [urlbase, basename, '.gz'];
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
    return
end

% Fall back to the raw uncompressed URL.
url_raw = [urlbase, basename];
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
