function [npass, nfail] = run_tests(varargin)
% RUN_TESTS  Unit tests for siamize.m. Works in MATLAB and GNU Octave.
%
%   [npass, nfail] = run_tests()
%   run_tests --exit            % command-line mode: exit(nfail) on completion
%
%   The tests stub the underlying `siamex` MEX so they only exercise the
%   pure-MATLAB dispatcher logic in siamize.m. The stub captures the
%   forwarded (img, affine, models, opts) and synthesizes a label volume
%   the same shape as img. This means the tests run in under a second and
%   require no ORT, no weight files, and no network.
%
%   Tests cover:
%       * argument-form dispatch (file / jnifti struct / readnifti struct
%         / bare array, with and without an affine)
%       * default centered-affine math when no affine is given
%       * model spec parsing (numeric, char digit, char CSV, cellstr,
%         mixed cellstr, single-digit fold shortcuts)
%       * file-in / file-out across .nii, .nii.gz, .jnii, .bnii
%       * 2-arg disambiguation (model spec vs output filename)
%       * error paths (bad first arg, out-of-range fold, bad cell, ...)

here = fileparts(mfilename('fullpath'));
matlab_dir = fileparts(here);
repo_root  = fileparts(matlab_dir);

% jsonlab first so bundled loadjd/loadnifti/jnii2nii/savejnifti/etc.
% shadow any system-wide install on the test runner's path.
addpath(fullfile(matlab_dir, 'jsonlab'), '-begin');
addpath(matlab_dir);

fx = setup_fixture_(matlab_dir);
cleanupObj = onCleanup(@() teardown_fixture_(fx));

tests = {
         @test_bare_array_defaults_, ...
         @test_bare_array_with_affine_3x4_, ...
         @test_bare_array_with_affine_4x4_, ...
         @test_jnifti_struct_with_affine_, ...
         @test_jnifti_struct_no_affine_default_, ...
         @test_readnifti_struct_sform_, ...
         @test_readnifti_struct_zeroed_sform_fallback_, ...
         @test_file_path_input_, ...
         @test_model_numeric_scalar_, ...
         @test_model_numeric_vector_, ...
         @test_model_char_digit_, ...
         @test_model_char_csv_, ...
         @test_model_cellstr_, ...
         @test_model_cellstr_mixed_, ...
         @test_model_empty_defaults_, ...
         @test_model_path_passthrough_, ...
         @test_default_affine_math_, ...
         @test_outputfile_nii_gz_, ...
         @test_outputfile_nii_uncompressed_, ...
         @test_outputfile_jnii_, ...
         @test_outputfile_bnii_, ...
         @test_outputfile_preserves_source_header_, ...
         @test_outputfile_synth_header_from_bare_array_, ...
         @test_disambiguate_digit_vs_filename_, ...
         @test_array_affine_outputfile_, ...
         @test_reject_bad_first_arg_, ...
         @test_reject_struct_missing_fields_, ...
         @test_reject_fold_out_of_range_, ...
         @test_reject_non_char_cell_entry_, ...
         @test_reject_unknown_output_extension_, ...
         @test_opts_struct_, ...
         @test_opts_name_value_pairs_, ...
         @test_opts_struct_plus_overrides_, ...
         @test_opts_cuda_tuning_passthrough_, ...
         @test_opts_gpu_passthrough_, ...
         @test_opts_tpm_t_passthrough_ ...
        };

do_exit = any(strcmp(varargin, '--exit'));
npass = 0;
nfail = 0;
fprintf('siamize.m unit tests\n');
fprintf('====================\n');
for k = 1:numel(tests)
    name = func2str(tests{k});
    % siamex stub state is per-test; reset between runs.
    clear -global SIAMEX_LAST;
    try
        tests{k}(fx);
        fprintf('  PASS  %s\n', name);
        npass = npass + 1;
    catch err
        fprintf('  FAIL  %s\n        %s\n', name, err.message);
        nfail = nfail + 1;
    end
end
fprintf('====================\n');
fprintf('summary: %d pass, %d fail (of %d)\n', npass, nfail, numel(tests));

if do_exit
    if exist('OCTAVE_VERSION', 'builtin') == 5
        exit(double(nfail > 0));
    else
        quit(double(nfail > 0));
    end
end
end

% =============================================================================
% Fixture
% =============================================================================

function fx = setup_fixture_(matlab_dir)
fx.matlab_dir = matlab_dir;

% Stub siamex so siamize.m's forwarding can be inspected without a
% real ORT/MEX. The stub writes to a global SIAMEX_LAST that tests
% then read back.
fx.stubdir = tempname();
mkdir(fx.stubdir);
stub = fullfile(fx.stubdir, 'siamex.m');
fid = fopen(stub, 'w');
fprintf(fid, [ ...
              'function lab = siamex(img, affine, models, opts)\n', ...
              '  global SIAMEX_LAST\n', ...
              '  SIAMEX_LAST.img    = img;\n', ...
              '  SIAMEX_LAST.affine = affine;\n', ...
              '  SIAMEX_LAST.models = models;\n', ...
              '  SIAMEX_LAST.opts   = opts;\n', ...
              '  lab = uint8(mod(double(img(:)), 18));\n', ...
              '  lab = reshape(lab, size(img));\n', ...
              'end\n']);
fclose(fid);
addpath(fx.stubdir, '-begin');

% Fake weight cache pre-populated with all 10 possible fold files so
% siamize_resolve_model_ short-circuits at the cache-hit branch.
fx.cachedir = tempname();
mkdir(fx.cachedir);
for k = 0:9
    f = fullfile(fx.cachedir, sprintf('fold_%d_fp16.onnx', k));
    fid = fopen(f, 'w');
    fprintf(fid, '.');
    fclose(fid);
end
fx.prev_cache_env = getenv('SIAMIZE_CACHE_DIR');
setenv('SIAMIZE_CACHE_DIR', fx.cachedir);

% Sample volume + affine reused across tests.
fx.img = uint8(reshape(mod(0:(20 * 30 * 40 - 1), 13), 20, 30, 40));
fx.A   = single([1.5, 0, 0, -15; 0, 1.5, 0, -20; 0, 0, 1.5, -25]);

% Pre-stage a .nii.gz fixture for the file-path test.
jnii = jnifticreate(fx.img);
jnii.NIFTIHeader.Affine = fx.A;
fx.sample_nii_gz = [tempname(), '.nii.gz'];
jnii2nii(jnii, fx.sample_nii_gz);

fx.tmpfiles = {fx.sample_nii_gz};
end

function teardown_fixture_(fx)
if isfield(fx, 'tmpfiles')
    for k = 1:numel(fx.tmpfiles)
        if exist(fx.tmpfiles{k}, 'file')
            delete(fx.tmpfiles{k});
        end
    end
end
if isfield(fx, 'cachedir') && exist(fx.cachedir, 'dir')
    rmdir(fx.cachedir, 's');
end
if isfield(fx, 'stubdir') && exist(fx.stubdir, 'dir')
    rmpath(fx.stubdir);
    rmdir(fx.stubdir, 's');
end
if isfield(fx, 'prev_cache_env')
    setenv('SIAMIZE_CACHE_DIR', fx.prev_cache_env);
end
end

% =============================================================================
% Argument-form tests
% =============================================================================

function test_bare_array_defaults_(fx)
global SIAMEX_LAST
lab = siamize(fx.img);
assert(isequal(size(lab), size(fx.img)), 'label shape mismatch');
A = SIAMEX_LAST.affine;
assert_centered_affine_(A, size(fx.img));
end

function test_bare_array_with_affine_3x4_(fx)
global SIAMEX_LAST
siamize(fx.img, fx.A);
assert(isequal(SIAMEX_LAST.affine, double(fx.A)), 'affine not forwarded');
end

function test_bare_array_with_affine_4x4_(fx)
global SIAMEX_LAST
A4 = single([fx.A; 0, 0, 0, 1]);
siamize(fx.img, A4);
assert(isequal(SIAMEX_LAST.affine, double(A4)), '4x4 affine not forwarded');
end

function test_jnifti_struct_with_affine_(fx)
global SIAMEX_LAST
nii = struct();
nii.NIFTIData = fx.img;
nii.NIFTIHeader = struct('Affine', fx.A);
siamize(nii);
assert(isequal(SIAMEX_LAST.img, fx.img), 'img not extracted');
assert(isequal(SIAMEX_LAST.affine, double(fx.A)), 'affine not extracted');
end

function test_jnifti_struct_no_affine_default_(fx)
global SIAMEX_LAST
nii = struct('NIFTIData', fx.img);
siamize(nii);
assert_centered_affine_(SIAMEX_LAST.affine, size(fx.img));
end

function test_readnifti_struct_sform_(fx)
global SIAMEX_LAST
nii.img = fx.img;
nii.hdr.srow_x = single([fx.A(1, :)]);
nii.hdr.srow_y = single([fx.A(2, :)]);
nii.hdr.srow_z = single([fx.A(3, :)]);
siamize(nii);
assert(isequal(SIAMEX_LAST.img, fx.img), 'img not extracted');
assert(max(abs(SIAMEX_LAST.affine(:) - double(fx.A(:)))) < 1e-5, ...
       'affine row-stacking mismatch');
end

function test_readnifti_struct_zeroed_sform_fallback_(fx)
global SIAMEX_LAST
nii.img = fx.img;
nii.hdr.srow_x = zeros(1, 4, 'single');
nii.hdr.srow_y = zeros(1, 4, 'single');
nii.hdr.srow_z = zeros(1, 4, 'single');
siamize(nii);
assert_centered_affine_(SIAMEX_LAST.affine, size(fx.img));
end

function test_file_path_input_(fx)
global SIAMEX_LAST
siamize(fx.sample_nii_gz);
assert(isequal(size(SIAMEX_LAST.img), size(fx.img)), ...
       'loaded shape mismatch: got %s', mat2str(size(SIAMEX_LAST.img)));
assert(max(abs(SIAMEX_LAST.affine(:) - double(fx.A(:)))) < 1e-4, ...
       'loaded affine mismatch');
end

% =============================================================================
% Model spec parsing
% =============================================================================

function test_model_numeric_scalar_(fx)
global SIAMEX_LAST
siamize(fx.img, fx.A, 3);
assert(numel(SIAMEX_LAST.models) == 1);
assert_basename_eq_(SIAMEX_LAST.models{1}, 'fold_3_fp16.onnx');
end

function test_model_numeric_vector_(fx)
global SIAMEX_LAST
siamize(fx.img, fx.A, [0, 2, 4]);
bs = basenames_(SIAMEX_LAST.models);
assert(isequal(bs, {'fold_0_fp16.onnx', 'fold_2_fp16.onnx', 'fold_4_fp16.onnx'}));
end

function test_model_char_digit_(fx)
global SIAMEX_LAST
siamize(fx.img, fx.A, '1');
assert_basename_eq_(SIAMEX_LAST.models{1}, 'fold_1_fp16.onnx');
end

function test_model_char_csv_(fx)
global SIAMEX_LAST
siamize(fx.img, fx.A, '0,1,2,3,4');
assert(numel(SIAMEX_LAST.models) == 5);
assert_basename_eq_(SIAMEX_LAST.models{1}, 'fold_0_fp16.onnx');
assert_basename_eq_(SIAMEX_LAST.models{5}, 'fold_4_fp16.onnx');
end

function test_model_cellstr_(fx)
global SIAMEX_LAST
siamize(fx.img, fx.A, {'0', '2', '4'});
bs = basenames_(SIAMEX_LAST.models);
assert(isequal(bs, {'fold_0_fp16.onnx', 'fold_2_fp16.onnx', 'fold_4_fp16.onnx'}));
end

function test_model_cellstr_mixed_(fx)
global SIAMEX_LAST
siamize(fx.img, fx.A, {'fold_1_fp16.onnx', '3'});
bs = basenames_(SIAMEX_LAST.models);
assert(isequal(bs, {'fold_1_fp16.onnx', 'fold_3_fp16.onnx'}));
end

function test_model_empty_defaults_(fx)
global SIAMEX_LAST
siamize(fx.img, fx.A, []);
assert(numel(SIAMEX_LAST.models) == 1);
assert_basename_eq_(SIAMEX_LAST.models{1}, 'fold_0_fp16.onnx');
end

function test_model_path_passthrough_(fx)
global SIAMEX_LAST
full = fullfile(fx.cachedir, 'fold_2_fp16.onnx');
siamize(fx.img, fx.A, full);
assert(strcmp(SIAMEX_LAST.models{1}, full), 'full path not preserved verbatim');
end

% =============================================================================
% Default-affine math
% =============================================================================

function test_default_affine_math_(fx)
global SIAMEX_LAST
sz = [40, 50, 60];
siamize(zeros(sz, 'uint8'));
A = SIAMEX_LAST.affine;
assert(isequal(A(1:3, 1:3), eye(3)), 'rotation should be identity');
expected = [-(sz(1) - 1) / 2, -(sz(2) - 1) / 2, -(sz(3) - 1) / 2];
assert(max(abs(A(:, 4)' - expected)) < 1e-9, ...
       'translation off: got %s want %s', ...
       mat2str(A(:, 4)'), mat2str(expected));
end

% =============================================================================
% File-in / file-out
% =============================================================================

function test_outputfile_nii_gz_(fx)
out = [tempname(), '.nii.gz'];
c = onCleanup(@() safe_delete_(out));
siamize(fx.sample_nii_gz, out);
assert(exist(out, 'file') == 2, 'output not written');
re = loadjd(out);
assert(isfield(re, 'NIFTIData'));
assert(isequal(size(re.NIFTIData), size(fx.img)));
end

function test_outputfile_nii_uncompressed_(fx)
out = [tempname(), '.nii'];
c = onCleanup(@() safe_delete_(out));
siamize(fx.sample_nii_gz, out);
assert(exist(out, 'file') == 2);
re = loadjd(out);
assert(isequal(size(re.NIFTIData), size(fx.img)));
end

function test_outputfile_jnii_(fx)
out = [tempname(), '.jnii'];
c = onCleanup(@() safe_delete_(out));
siamize(fx.sample_nii_gz, out);
assert(exist(out, 'file') == 2);
re = loadjd(out);
assert(isfield(re, 'NIFTIData'));
assert(isequal(size(re.NIFTIData), size(fx.img)));
end

function test_outputfile_bnii_(fx)
out = [tempname(), '.bnii'];
c = onCleanup(@() safe_delete_(out));
siamize(fx.sample_nii_gz, out);
assert(exist(out, 'file') == 2);
re = loadjd(out);
assert(isfield(re, 'NIFTIData'));
assert(isequal(size(re.NIFTIData), size(fx.img)));
end

function test_outputfile_preserves_source_header_(fx)
out = [tempname(), '.nii.gz'];
c = onCleanup(@() safe_delete_(out));
siamize(fx.sample_nii_gz, out);
re = loadjd(out);
assert(max(abs(double(re.NIFTIHeader.Affine(:)) - double(fx.A(:)))) < 1e-4, ...
       'source affine not preserved on round-trip');
end

function test_outputfile_synth_header_from_bare_array_(fx)
out = [tempname(), '.nii.gz'];
c = onCleanup(@() safe_delete_(out));
siamize(fx.img, out, 0);
assert(exist(out, 'file') == 2);
re = loadjd(out);
assert(isequal(size(re.NIFTIData), size(fx.img)));
A = double(re.NIFTIHeader.Affine);
expected = [-(size(fx.img, 1) - 1) / 2, ...
            -(size(fx.img, 2) - 1) / 2, ...
            -(size(fx.img, 3) - 1) / 2];
assert(max(abs(A(:, 4)' - expected)) < 1e-4, ...
       'synth header affine not centered');
end

function test_disambiguate_digit_vs_filename_(fx)
global SIAMEX_LAST
% '0' is not a filename ext -> falls through to model parser.
siamize(fx.sample_nii_gz, '0');
assert(numel(SIAMEX_LAST.models) == 1);
assert_basename_eq_(SIAMEX_LAST.models{1}, 'fold_0_fp16.onnx');
end

function test_array_affine_outputfile_(fx)
out = [tempname(), '.nii.gz'];
c = onCleanup(@() safe_delete_(out));
siamize(fx.img, fx.A, out, 0);
assert(exist(out, 'file') == 2);
re = loadjd(out);
assert(isequal(size(re.NIFTIData), size(fx.img)));
end

% =============================================================================
% Error paths
% =============================================================================

function test_reject_bad_first_arg_(fx)  %#ok<INUSD>
err = catch_call_(@() siamize(true));
assert(~isempty(strfind(err.message, 'first input must be')), ...
       'wrong error: %s', err.message);
end

function test_reject_struct_missing_fields_(fx)  %#ok<INUSD>
err = catch_call_(@() siamize(struct('foo', 1)));
assert(~isempty(strfind(err.message, 'jnifti')), ...
       'wrong error: %s', err.message);
end

function test_reject_fold_out_of_range_(fx)
err = catch_call_(@() siamize(fx.img, fx.A, 11));
assert(~isempty(strfind(err.message, 'integer 0..9')), ...
       'wrong error: %s', err.message);
end

function test_reject_non_char_cell_entry_(fx)
err = catch_call_(@() siamize(fx.img, fx.A, {123}));
assert(~isempty(strfind(err.message, 'must be a char')), ...
       'wrong error: %s', err.message);
end

function test_reject_unknown_output_extension_(fx)
% The image-extension regex doesn't match .txt, so 'out.txt' is parsed
% as a model spec. The model parser then tries to use 'out.txt' as a
% single-fold filename -> resolve_model_path fetch fails (no .gz, no
% .onnx URL). Expect a fetch / resolve error, NOT an output-write one.
err = catch_call_(@() siamize(fx.img, 'out.txt'));
assert(~isempty(err.message), 'expected some failure');
end

% =============================================================================
% Opts plumbing (varargin2struct pattern)
% =============================================================================

function test_opts_struct_(fx)
global SIAMEX_LAST
siamize(fx.img, fx.A, 0, struct('compute', 'cpu', 'verbose', true));
assert(strcmp(SIAMEX_LAST.opts.compute, 'cpu'));
assert(logical(SIAMEX_LAST.opts.verbose));
end

function test_opts_name_value_pairs_(fx)
global SIAMEX_LAST
siamize(fx.img, fx.A, 0, 'compute', 'cpu', 'verbose', true);
assert(strcmp(SIAMEX_LAST.opts.compute, 'cpu'));
assert(logical(SIAMEX_LAST.opts.verbose));
end

function test_opts_struct_plus_overrides_(fx)
global SIAMEX_LAST
defs = struct('compute', 'auto', 'verbose', false);
siamize(fx.img, fx.A, 0, defs, 'verbose', true);
% 'verbose' override should win; 'compute' should stay from struct.
assert(strcmp(SIAMEX_LAST.opts.compute, 'auto'));
assert(logical(SIAMEX_LAST.opts.verbose));
end

function test_opts_cuda_tuning_passthrough_(fx)
global SIAMEX_LAST
siamize(fx.img, fx.A, 0, ...
        'cudnn_max_workspace', 0, ...
        'arena_extend', 'same', ...
        'cudnn_algo', 'heuristic', ...
        'gpu_mem_limit', 6 * 1024^3);
assert(SIAMEX_LAST.opts.cudnn_max_workspace == 0);
assert(strcmp(SIAMEX_LAST.opts.arena_extend, 'same'));
assert(strcmp(SIAMEX_LAST.opts.cudnn_algo, 'heuristic'));
assert(SIAMEX_LAST.opts.gpu_mem_limit == 6 * 1024^3);
end

function test_opts_gpu_passthrough_(fx)
global SIAMEX_LAST
siamize(fx.img, fx.A, 0, 'gpu', 2);
assert(SIAMEX_LAST.opts.gpu == 2);
end

function test_opts_tpm_t_passthrough_(fx)
global SIAMEX_LAST
siamize(fx.img, fx.A, 0, 'tpm_t', 1.5);
assert(abs(SIAMEX_LAST.opts.tpm_t - 1.5) < 1e-9);
end

% =============================================================================
% Helpers
% =============================================================================

function assert_centered_affine_(A, sz)
assert(isequal(size(A), [3, 4]) || isequal(size(A), [4, 4]), ...
       'affine wrong shape: %s', mat2str(size(A)));
assert(isequal(A(1:3, 1:3), eye(3)), 'rotation should be identity');
expected = [-(double(sz(1)) - 1) / 2, ...
            -(double(sz(2)) - 1) / 2, ...
            -(double(sz(3)) - 1) / 2];
assert(max(abs(A(1:3, 4)' - expected)) < 1e-9, ...
       'centered translation off: got %s want %s', ...
       mat2str(A(1:3, 4)'), mat2str(expected));
end

function assert_basename_eq_(p, want)
[~, n, e] = fileparts(p);
got = [n, e];
assert(strcmp(got, want), 'basename mismatch: got %s want %s', got, want);
end

function bs = basenames_(paths)
bs = cell(1, numel(paths));
for k = 1:numel(paths)
    [~, n, e] = fileparts(paths{k});
    bs{k} = [n, e];
end
end

function err = catch_call_(fn)
try
    fn();
    err = struct('message', '<no error thrown>');
    error('siamize:test', 'expected exception');
catch err
    if strcmp(err.identifier, 'siamize:test')
        rethrow(err);
    end
end
end

function safe_delete_(p)
if exist(p, 'file')
    delete(p);
end
end
