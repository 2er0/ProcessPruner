import numpy as np
import pandas as pd
from tqdm import tqdm


# calculate the similarity
def get_fitness(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    size = len(a)
    return 1 - ((size - np.sum(a == b)) / size)


# calculate the similarity between each footprint matrix
# return n most similar matrices
def get_similar_footprints(amount, mats, starts, ends):
    if amount < len(mats):
        dim = len(mats)
        fitt_mat = np.zeros((dim, dim))

        for x in range(0, dim):
            for y in range(0, dim):
                fitt_mat[x, y] = get_fitness(mats[x], mats[y])

        similarity_map = []

        for i in range(0, dim):
            similarity_map.append((i, sum(fitt_mat[i, :])))

        similarity_map.sort(key=lambda x: x[1])

        sim_idx = [x[0] for x in similarity_map[-amount:]]
        avg_sim = np.mean([x[1] for x in similarity_map[-amount:]]) / dim

        return mats[sim_idx], starts[sim_idx], ends[sim_idx], avg_sim
    else:
        raise ValueError('Amount of result matrizes has to be lower than input matrizes amount!')


def process_pruner(full_log=False):
    # what is the csv source
    source = input('Type the path and name of the source CSV-file: ').strip()
    output = input('Type the path and name for the outputs without extension: ').strip()

    # what is the separator for the csv
    sep = input('Separator for the input CSV: ')
    log = pd.read_csv(f"{source}", sep=sep)

    # what is the Case/Trace identifier
    print('\n')
    for i, val in enumerate(log.columns.values):
        print('[{}]: {}'.format(i, val))
    case_ident = int(input('Type number for Case/Trace identifier: '))
    case_ident = str(log.columns.values[case_ident])

    # ask user what is the Activity/Transaction identifier
    print('\n')
    for i, val in enumerate(log.columns.values):
        print('[{}]: {}'.format(i, val))
    activity = int(input('Type number for Activity/Transaction identifier: '))
    activity = str(log.columns.values[activity])

    # get all activities
    act = list(set(log[activity]))

    end = set()

    # get all ends
    print('Finding each end event of each trace ...')
    for trace in tqdm(log.groupby(case_ident)):
        end.add(trace[1].tail(1)[activity].values[0])
    end = list(end)

    # ask user what should not be allowed as end, 'enter' keeps all
    # get no end identifiers for drop
    print('\n')
    for i, e in enumerate(end):
        print('[{}]: {}'.format(i, e))

    end_nodes = input('Type numbers of NO end transactions like "0,1" or pres `enter` to continue: ')
    if len(end_nodes) > 0:
        end_nodes = [int(n) for n in end_nodes.split(',')]
        end = [e for i, e in enumerate(end) if i not in end_nodes]

    # show remaining ends
    print('\nselected ends: {}'.format(end))

    # get case ids for dropping
    print('Executing end event filtering if defined ...')
    ids_to_drop = [trace[0] for trace in tqdm(log.groupby(case_ident))
                   if trace[1].tail(1)[activity].values[0] not in end]

    # show how many traces will be dropped by not right ending
    print('Number of traces that will be dropped by not fulfilling the end criteria: {}'.format(len(ids_to_drop)))

    # drop traces
    log.drop(log[log[case_ident].isin(ids_to_drop)].index, inplace=True)

    # prepare for many logic
    mats = []
    starts = []
    ends = []
    log_back = log.copy(deep=True)

    # ask user how many traces should be used for generating one matrix
    traces_to_use = 30
    traces_to_use_user = input('\nType the number of traces to use for each footprint or press enter: (30=default) ')
    if len(traces_to_use_user) > 0:
        traces_to_use = int(traces_to_use_user)

    print('Main analysis started - matrices are now generated and analysed ...')

    # shuffle all ids to not favor anything
    unique_ids = log[case_ident].unique()
    np.random.shuffle(unique_ids)

    # calculate the number of iterations needed for processing each trace once
    full_log_size = len(unique_ids)
    runs = int(full_log_size / traces_to_use)
    over = 0 if full_log_size % traces_to_use == 0 else 1
    runs = runs + over

    # generate N footprint matrices
    for _ in tqdm(range(runs)):

        mat = np.full((len(act), len(act)), ['#'], dtype=str)

        start = []
        end = []

        sample_size = traces_to_use if len(unique_ids) >= traces_to_use else len(unique_ids)

        sub = unique_ids[:sample_size]
        unique_ids = unique_ids[sample_size:]

        # select next K unique traces
        road_groups = log.loc[log[case_ident].isin(sub)].groupby(case_ident)

        # generate footprint matrix
        for trace in road_groups:
            init = False
            prev = -1
            last = ''
            for next in trace[1].iterrows():
                if not init:
                    init = True
                    prev = act.index(next[1][activity])
                    # find start
                    if start.count(next[1][activity]) == 0:
                        start.append(next[1][activity])
                else:
                    cur = act.index(next[1][activity])

                    if not mat[prev, cur] in ['<', '|']:
                        mat[prev, cur] = '>'
                        mat[cur, prev] = '<'
                    else:
                        mat[prev, cur] = '|'
                        mat[cur, prev] = '|'

                    prev = cur
                    last = next[1][activity]

            # find end
            if end.count(last) == 0:
                end.append(last)

        mats.append(mat)
        starts.append(start)
        ends.append(end)

        # drop used traces
        log.drop(log[log[case_ident].isin(sub)].index, inplace=True)

    print(len(mats))
    mats = np.asarray(mats)
    starts = np.asarray(starts)
    ends = np.asarray(ends)

    # calculate and get most similar footprint matrices or the one which is most similar to others
    main_mat, starts, ends, avgSim = get_similar_footprints(1, mats, starts, ends)
    # get the first matrix which should be the most general one
    main_mat = main_mat[0]
    main_start = starts[0]
    main_end = ends[0]
    print('sim: {} | starts: {} | ends: {}'.format(avgSim, main_start, main_end))

    # replay all traces against the selected matrix
    match_results = []
    match_output= []
    match_remove = []
    for trace in tqdm(log_back.groupby(case_ident)):
        init = False
        prev = -1
        last = ''
        start_found = False
        count = 0
        for next in trace[1].iterrows():
            if not init:
                init = True
                prev = act.index(next[1][activity])
                begin = next[1][activity]
                # is start correct
                start_found = begin in main_start

            else:
                cur = act.index(next[1][activity])

                if main_mat[prev, cur] in ['>', '|']:
                    count += 1

                prev = cur
                last = next[1][activity]

        # find end
        end_found = last in main_end

        match_value = count / (len(trace[1]) - 1)

        to_print = 'Case ID: {} | start_match: {} | end_match: {} | match_factor: {}' \
            .format(trace[0], start_found, end_found, match_value)

        # set the level of replay ability
        # 1 means 100%
        # the selected start and end form the chosen matrix must also match
        if match_value < 1 or not start_found or not end_found:
            match_remove.append(trace[0])
        else:
            match_results.append(trace[0])
            match_output.append(to_print)

    if full_log:
        for i in match_results:
            print(i)

    print('sim: {} | starts: {} | ends: {}'.format(avgSim, main_start, main_end))
    print('ori {} vs {} filtered'.format(len(log_back[case_ident].unique()), len(match_results)))

    log_back_no_match = log_back.copy(deep=True)
    log_back.drop(log_back[log_back[case_ident].isin(match_remove)].index, inplace=True)
    log_back.to_csv(f'{output}-match.csv', index=False, sep=';')
    print(f'Matching traces are saved to {output}-match.csv')

    log_back_no_match.drop(log_back_no_match[log_back_no_match[case_ident].isin(match_results)].index, inplace=True)
    log_back_no_match.to_csv(f'{output}-no-match.csv', index=False, sep=';')
    print(f'No matching traces are saved to {output}-no-match.csv')

    mat = pd.DataFrame(main_mat)
    mat.columns = act
    tmp = {}
    for x, v in enumerate(act):
        tmp[x] = v
    mat = mat.rename(index=tmp)
    print(mat.to_latex())


if __name__ == "__main__":
    process_pruner()
