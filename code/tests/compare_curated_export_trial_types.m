%COMPARE_CURATED_EXPORT_TRIAL_TYPES Compare curated vs export trialinfo labels.
%
% CSV columns:
%   n_trials_total_*     — all trials in session (length of TrialSubType_list)
%   {Type}_rewarded_*    — trials of that subtype with RA1-4 or RB1-4 only
%   '-'                  — no trials of that subtype in the session
%
% Usage (S: mounted):
%   cd('.../BoS-ephys-MUA/code/tests')
%   compare_curated_export_trial_types
%   compare_curated_export_trial_types(5)
%   compare_curated_export_trial_types(10, outCsvPath)

function compare_curated_export_trial_types(nSessions, outCsv)
    if nargin < 1 || isempty(nSessions)
        nSessions = 10;
    end

    repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    cd(repoRoot);
    session_lists; %#ok<RUNSESSION> % loads root_folder, *_CONF cell arrays

    if nargin < 2 || isempty(outCsv)
        outCsv = fullfile(repoRoot, 'figures', 'debug', 'curated_vs_export_trial_types.csv');
    end

    curatedRoot = 'S:\taskcontroller\SCP_DATA\SCP-CTRL-01\MUA_curated_sessions';
    conditions = {'Curius_BLOCKED', 'Curius_SHUFFLED', 'Elmo_BLOCKED', 'Elmo_SHUFFLED'};
    trialTypes = {'Dyadic', 'SemiSolo', 'SoloA', 'SoloARewardAB', 'SoloB', 'SoloBRewardAB'};
    rewardedA = {'RA1', 'RA2', 'RA3', 'RA4'};
    rewardedB = {'RB1', 'RB2', 'RB3', 'RB4'};

    rows = {};
    for c = 1:numel(conditions)
        condition = conditions{c};
        confName = [condition '_CONF'];
        confIds = eval(confName); %#ok<EVLDIR>
        listMonkey = list_monkey_from_condition(condition);

        curatedIds = discover_curated_sessions(curatedRoot, condition);
        sessionIds = pick_curated_in_conf(curatedIds, confIds, nSessions);

        fprintf('=== %s (%d sessions) ===\n', condition, numel(sessionIds));
        for s = 1:numel(sessionIds)
            sessionId = sessionIds{s};
            prefix = strtok(sessionId, '.');
            curatedDir = fullfile(curatedRoot, condition, sessionId);
            exportDir = fullfile(root_folder, sessionId);

            fprintf('  [%d/%d] %s ... ', s, numel(sessionIds), prefix);
            t0 = tic;

            labelsCur = load_trial_labels(curatedDir, sessionId);
            labelsExp = load_trial_labels(exportDir, sessionId);

            nTrialsCur = label_list_length(labelsCur);
            nTrialsExp = label_list_length(labelsExp);

            countsCur = rewarded_type_counts(labelsCur, listMonkey, trialTypes, rewardedA, rewardedB);
            countsExp = rewarded_type_counts(labelsExp, listMonkey, trialTypes, rewardedA, rewardedB);

            typeMatch = strcmp(countsCur, countsExp);
            matchAll = all(typeMatch) && (nTrialsCur == nTrialsExp);

            row = {condition, prefix, sessionId, nTrialsCur, nTrialsExp};
            for t = 1:numel(trialTypes)
                row{end+1} = countsCur{t}; %#ok<AGROW>
                row{end+1} = countsExp{t}; %#ok<AGROW>
                row{end+1} = match_label(typeMatch(t)); %#ok<AGROW>
            end
            row{end+1} = match_label(matchAll);
            rows(end+1, 1:numel(row)) = row; %#ok<AGROW>

            fprintf('%.1fs match=%s\n', toc(t0), row{end});
        end
    end

    header = {'condition', 'session', 'session_id', ...
        'n_trials_total_curated', 'n_trials_total_export'};
    for t = 1:numel(trialTypes)
        tt = trialTypes{t};
        header = [header, {[tt '_rewarded_curated'], [tt '_rewarded_export'], [tt '_rewarded_match']}];
    end
    header = [header, {'match_all'}];

    csvComment = ['# n_trials_total_* = all trials in session; ', ...
        '{Type}_rewarded_* = that subtype with RA1-4/RB1-4 only; ', ...
        '"-" = subtype absent'];

    outDir = fileparts(outCsv);
    if ~isempty(outDir) && ~isfolder(outDir)
        mkdir(outDir);
    end

    write_csv(outCsv, header, rows, csvComment);
    fprintf('\nWrote %s (%d rows)\n', outCsv, size(rows, 1));
end

function ids = discover_curated_sessions(curatedRoot, condition)
    d = dir(fullfile(curatedRoot, condition));
    d = d([d.isdir] & ~ismember({d.name}, {'.', '..'}));
    ids = sort({d.name})';
end

function picked = pick_curated_in_conf(curatedIds, confIds, n)
    confSet = string(confIds(:));
    picked = {};
    for i = 1:numel(curatedIds)
        sid = curatedIds{i};
        if any(confSet == string(sid))
            picked{end+1} = sid; %#ok<AGROW>
            if numel(picked) >= n
                return;
            end
        end
    end
    error('compare_curated_export_trial_types:only %d/%d curated sessions in CONF list', ...
        numel(picked), n);
end

function monkey = list_monkey_from_condition(condition)
    if startsWith(condition, 'Elmo')
        monkey = 'Elmo';
    elseif startsWith(condition, 'Curius')
        monkey = 'Curius';
    else
        error('Unknown condition: %s', condition);
    end
end

function labels = load_trial_labels(sessionDir, sessionId)
    path = fullfile(sessionDir, [sessionId '.trialinfo.4python.mat']);
    if ~isfile(path)
        error('Missing trialinfo: %s', path);
    end
    S = load(path);
    labels = S.cur_raster_labels;
end

function n = label_list_length(labels)
    if ~isfield(labels, 'TrialSubType_list')
        error('TrialSubType_list missing');
    end
    n = numel(labels.TrialSubType_list);
end

function counts = rewarded_type_counts(labels, listMonkey, trialTypes, rewardedA, rewardedB)
    subtypes = cellstr_labels(labels.TrialSubType_list);
    counts = cell(size(trialTypes));
    for i = 1:numel(trialTypes)
        tt = trialTypes{i};
        typeMask = strcmp(subtypes, tt);
        if ~any(typeMask)
            counts{i} = '-';
            continue;
        end
        rewardField = reward_field_for_type(tt, listMonkey);
        if ~isfield(labels, rewardField)
            counts{i} = '-';
            continue;
        end
        rewards = cellstr_labels(labels.(rewardField));
        if strcmp(rewardField, 'A_Reward_list')
            rewardedMask = ismember(rewards, rewardedA);
        else
            rewardedMask = ismember(rewards, rewardedB);
        end
        counts{i} = num2str(sum(typeMask & rewardedMask));
    end
end

function field = reward_field_for_type(trialType, listMonkey)
    switch trialType
        case {'SoloA', 'SoloARewardAB'}
            field = 'A_Reward_list';
        case {'SoloB', 'SoloBRewardAB'}
            field = 'B_Reward_list';
        case {'Dyadic', 'SemiSolo'}
            if strcmp(listMonkey, 'Curius')
                field = 'A_Reward_list';
            else
                field = 'B_Reward_list';
            end
        otherwise
            error('Unknown trial type: %s', trialType);
    end
end

function out = cellstr_labels(x)
    if isstring(x)
        out = cellstr(x(:));
    elseif iscell(x)
        out = x(:);
    else
        out = cellstr(x);
        out = out(:);
    end
    for i = 1:numel(out)
        if isstring(out{i})
            out{i} = char(out{i});
        end
    end
end

function s = match_label(tf)
    if tf
        s = 'yes';
    else
        s = 'NO';
    end
end

function write_csv(path, header, rows, comment)
    fid = fopen(path, 'w');
    if fid < 0
        error('Cannot write %s', path);
    end
    cleaner = onCleanup(@() fclose(fid));
    if nargin >= 4 && ~isempty(comment)
        fprintf(fid, '%s\n', comment);
    end
    for i = 1:numel(header) - 1
        fprintf(fid, '%s,', header{i});
    end
    fprintf(fid, '%s\n', header{end});
    for r = 1:size(rows, 1)
        row = rows(r, :);
        for c = 1:numel(header) - 1
            fprintf(fid, '%s,', csv_cell(row{c}));
        end
        fprintf(fid, '%s\n', csv_cell(row{end}));
    end
end

function s = csv_cell(v)
    if isnumeric(v)
        s = num2str(v);
    else
        s = char(string(v));
    end
end
