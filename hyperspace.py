import itertools
import copy

from constants import *
from pipeline.base import get_pipeline

from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from ConfigSpace.forbidden import ForbiddenAndConjunction, ForbiddenEqualsClause
from ConfigSpace.hyperparameters import NumericalHyperparameter

import numpy as np


def check_pipeline(pipeline, data_info, include=None, exclude=None):
    sparse = data_info["is_sparse"]
    signed = data_info["signed"]

    # FIXME structural optimization
    # Duck typing, not sure if it's good...
    node_i_is_choice = []
    node_i_choices = []
    node_i_choices_names = []
    all_nodes = []
    for node_name, node in pipeline:
        all_nodes.append(node)
        is_choice = hasattr(node, "get_available_components")
        node_i_is_choice.append(is_choice)

        node_include = include.get(
            node_name) if include is not None else None
        node_exclude = exclude.get(
            node_name) if exclude is not None else None

        if is_choice:
            node_i_choices_names.append(list(node.get_available_components(
                data_info, include=node_include, exclude=node_exclude).keys()))
            node_i_choices.append(list(node.get_available_components(
                data_info, include=node_include, exclude=node_exclude).values()))

        else:
            node_i_choices.append([node])

    matches_dimensions = [len(choices) for choices in node_i_choices]
    # Start by allowing every combination of nodes. Go through all
    # combinations/pipelines and erase the illegal ones
    matches = np.ones(matches_dimensions, dtype=int)

    pipeline_idxs = [range(dim) for dim in matches_dimensions]
    for pipeline_instantiation_idxs in itertools.product(*pipeline_idxs):
        pipeline_instantiation = [node_i_choices[i][idx] for i, idx in
                                  enumerate(pipeline_instantiation_idxs)]

        data_is_sparse = sparse
        dataset_is_signed = signed
        for node in pipeline_instantiation:
            node_input = node.get_properties()['input']
            node_output = node.get_properties()['output']

            # First check if these two instantiations of this node can work
            # together. Do this in multiple if statements to maintain
            # readability
            if data_is_sparse and (SPARSE not in node_input):
                matches[pipeline_instantiation_idxs] = 0
                break

            if not data_is_sparse and (DENSE not in node_input):
                matches[pipeline_instantiation_idxs] = 0
                break
            # No need to check if the node can handle SIGNED_DATA; this is
            # always assumed to be true
            if not dataset_is_signed and (UNSIGNED_DATA not in node_input):
                matches[pipeline_instantiation_idxs] = 0
                break

            if (INPUT in node_output and DENSE not in node_output and SPARSE not in node_output) or \
                    (PREDICTIONS in node_output) or \
                    (not data_is_sparse and DENSE in node_input and DENSE in node_output) or \
                    (data_is_sparse and SPARSE in node_input and SPARSE in node_output):
                # Don't change the data_is_sparse flag
                pass
            elif data_is_sparse and DENSE in node_output:
                data_is_sparse = False
            elif not data_is_sparse and SPARSE in node_output:
                data_is_sparse = True
            else:
                print(node)
                print("Data is sparse", data_is_sparse)
                print(node_input, node_output)
                raise ValueError("This combination is not allowed!")

            if PREDICTIONS in node_output:
                pass
            elif (INPUT in node_output and SIGNED_DATA not in node_output and
                        UNSIGNED_DATA not in node_output):
                pass
            elif SIGNED_DATA in node_output:
                dataset_is_signed = True
            elif UNSIGNED_DATA in node_output:
                dataset_is_signed = False
            else:
                print(node)
                print("Data is signed", dataset_is_signed)
                print(node_input, node_output)
                raise ValueError("This combination is not allowed!")

    return matches


def find_active_choices(matches, node, node_idx, data_info,
                        include=None, exclude=None):
    if not hasattr(node, "get_available_components"):
        raise ValueError()
    available_components = node.get_available_components(data_info,
                                                         include=include,
                                                         exclude=exclude)
    assert matches.shape[node_idx] == len(available_components), \
        (matches.shape[node_idx], len(available_components))

    choices = []
    for c_idx, component in enumerate(available_components):
        slices = [slice(None) if idx != node_idx else slice(c_idx, c_idx+1)
                  for idx in range(len(matches.shape))]

        if np.sum(matches[slices]) > 0:
            choices.append(component)
    return choices


def add_forbidden(conf_space, pipeline, matches, dataset_properties,
                  include=None, exclude=None):
    # Not sure if this works for 3D
    # FIXME Not works
    node_i_is_choice = []
    node_i_choices_names = []
    node_i_choices = []
    all_nodes = []
    for node_name, node in pipeline:
        all_nodes.append(node)
        is_choice = hasattr(node, "get_available_components")
        node_i_is_choice.append(is_choice)

        node_include = include.get(
            node_name) if include is not None else None
        node_exclude = exclude.get(
            node_name) if exclude is not None else None

        if is_choice:
            node_i_choices_names.append(node.get_available_components(
                dataset_properties, include=node_include,
                exclude=node_exclude).keys())
            node_i_choices.append(node.get_available_components(
                dataset_properties, include=node_include,
                exclude=node_exclude).values())

        else:
            node_i_choices_names.append([node_name])
            node_i_choices.append([node])

    # Find out all chains of choices. Only in such a chain its possible to
    # have several forbidden constraints
    choices_chains = []
    idx = 0
    while idx < len(pipeline):
        if node_i_is_choice[idx]:
            chain_start = idx
            idx += 1
            while idx < len(pipeline) and node_i_is_choice[idx]:
                idx += 1
            chain_stop = idx
            choices_chains.append((chain_start, chain_stop))
        idx += 1

    for choices_chain in choices_chains:
        constraints = set()

        chain_start = choices_chain[0]
        chain_stop = choices_chain[1]
        chain_length = chain_stop - chain_start

        # Add one to have also have chain_length in the range
        for sub_chain_length in range(2, chain_length + 1):
            for start_idx in range(chain_start, chain_stop - sub_chain_length + 1):
                indices = range(start_idx, start_idx + sub_chain_length)
                node_names = [pipeline[idx][0] for idx in indices]

                num_node_choices = []
                node_choice_names = []
                skip_array_shape = []

                for idx in indices:
                    node = all_nodes[idx]
                    available_components = node.get_available_components(
                        dataset_properties,
                        include=node_i_choices_names[idx])
                    assert len(available_components) > 0, len(available_components)
                    skip_array_shape.append(len(available_components))
                    num_node_choices.append(range(len(available_components)))
                    node_choice_names.append([name for name in available_components])

                # Figure out which choices were already abandoned
                skip_array = np.zeros(skip_array_shape)
                for product in itertools.product(*num_node_choices):
                    for node_idx, choice_idx in enumerate(product):
                        node_idx += start_idx
                        slices_ = [
                            slice(None) if idx != node_idx else
                            slice(choice_idx, choice_idx + 1) for idx in
                            range(len(matches.shape))]

                        if np.sum(matches[slices_]) == 0:
                            skip_array[product] = 1

                for product in itertools.product(*num_node_choices):
                    if skip_array[product]:
                        continue

                    slices = []
                    for idx in range(len(matches.shape)):
                        if idx not in indices:
                            slices.append(slice(None))
                        else:
                            slices.append(slice(product[idx - start_idx],
                                                product[idx - start_idx] + 1))

                    # This prints the affected nodes
                    # print [node_choice_names[i][product[i]]
                    #     for i in range(len(product))], \
                    #     np.sum(matches[slices])

                    if np.sum(matches[slices]) == 0:
                        constraint = tuple([(node_names[i],
                                             node_choice_names[i][product[i]])
                                            for i in range(len(product))])

                        # Check if a more general constraint/forbidden clause
                        #  was already added
                        continue_ = False
                        for constraint_length in range(2, len(constraint)):
                            for constraint_start_idx in range(len(constraint)
                                    - constraint_length + 1):
                                sub_constraint = constraint[
                                                     constraint_start_idx:constraint_start_idx + constraint_length]
                                if sub_constraint in constraints:
                                    continue_ = True
                                    break
                            if continue_:
                                break
                        if continue_:
                            continue

                        constraints.add(constraint)

                        forbiddens = []
                        for i in range(len(product)):
                            forbiddens.append(
                                ForbiddenEqualsClause(conf_space.get_hyperparameter(
                                    node_names[i] + ":__choice__"),
                                    node_choice_names[i][product[i]]))
                        forbidden = ForbiddenAndConjunction(*forbiddens)
                        conf_space.add_forbidden_clause(forbidden)

    return conf_space


def get_hyperspace(data_info,
                   include_estimators=None, include_preprocessors=None):

    if data_info is None or not isinstance(data_info, dict):
        data_info = dict()

    if 'is_sparse' not in data_info:
        # This dataset is probaby dense
        data_info['is_sparse'] = False

    sparse = data_info['is_sparse']
    task_type = data_info['task']
    multilabel = (task_type == MULTILABEL_CLASSIFICATION)
    multiclass = (task_type == MULTICLASS_CLASSIFICATION)

    if task_type in CLASSIFICATION_TASKS:
        data_info['multilabel'] = multilabel
        data_info['multiclass'] = multiclass
        data_info['target_type'] = 'classification'
        pipe_type = 'classifier'

        # Components match to be forbidden
        components_ = ["adaboost", "decision_tree", "extra_trees",
                    "gradient_boosting", "k_nearest_neighbors",
                    "libsvm_svc", "random_forest", "gaussian_nb",
                    "decision_tree"]
        feature_learning_ = ["kitchen_sinks", "nystroem_sampler"]
    elif task_type in REGRESSION_TASKS:
        data_info['target_type'] = 'regression'
        pipe_type = 'regressor'

        # Components match to be forbidden
        components_ = ["adaboost", "decision_tree", "extra_trees",
                       "gaussian_process", "gradient_boosting",
                       "k_nearest_neighbors", "random_forest"]
        feature_learning_ = ["kitchen_sinks", "kernel_pca", "nystroem_sampler"]
    else:
        raise NotImplementedError()

    include, exclude = dict(), dict()
    if include_preprocessors is not None:
        include["preprocessor"] = include_preprocessors
    if include_estimators is not None:
        include[pipe_type] = include_estimators

    cs = ConfigurationSpace()

    # Construct pipeline
    # FIXME OrderedDIct?
    pipeline = get_pipeline(data_info['task'])

    # TODO include, exclude, pipeline
    keys = [pair[0] for pair in pipeline]
    for key in include:
        if key not in keys:
            raise ValueError('Invalid key in include: %s; should be one '
                             'of %s' % (key, keys))

    for key in exclude:
            if key not in keys:
                raise ValueError('Invalid key in exclude: %s; should be one '
                                 'of %s' % (key, keys))

    # Construct hyperspace
    # TODO What's the 'signed' stands for?
    if 'signed' not in data_info:
        # This dataset probably contains unsigned data
        data_info['signed'] = False

    match = check_pipeline(pipeline, data_info,
                           include=include, exclude=exclude)

    # Now we have only legal combinations at this step of the pipeline
    # Simple sanity checks
    assert np.sum(match) != 0, "No valid pipeline found."

    assert np.sum(match) <= np.size(match), \
        "'matches' is not binary; %s <= %d, %s" % \
        (str(np.sum(match)), np.size(match), str(match.shape))

    # Iterate each dimension of the matches array (each step of the
    # pipeline) to see if we can add a hyperparameter for that step
    for node_idx, n_ in enumerate(pipeline):
        node_name, node = n_
        is_choice = hasattr(node, "get_available_components")

        # if the node isn't a choice we can add it immediately because it
        #  must be active (if it wouldn't, np.sum(matches) would be zero
        if not is_choice:
            cs.add_configuration_space(node_name,
                node.get_hyperparameter_search_space(data_info))
        # If the node isn't a choice, we have to figure out which of it's
        #  choices are actually legal choices
        else:
            choices_list = find_active_choices(match, node, node_idx,data_info,
                                               include=include.get(node_name),
                                               exclude=exclude.get(node_name))
            cs.add_configuration_space(node_name,
                node.get_hyperparameter_search_space(data_info,
                                                     include=choices_list))
    # And now add forbidden parameter configurations
    # According to matches
    if np.sum(match) < np.size(match):
        cs = add_forbidden(conf_space=cs, pipeline=pipeline, matches=match,
                           dataset_properties=data_info, include=include, exclude=exclude)

    components = cs.get_hyperparameter('%s:__choice__' % pipe_type).choices
    availables = pipeline[-1][1].get_available_components(data_info)

    preprocessors = cs.get_hyperparameter('preprocessor:__choice__').choices
    #available_preprocessors = pipeline[-2][1].get_available_components(data_info)


    possible_default = copy.copy(list(availables.keys()))
    default = cs.get_hyperparameter('%s:__choice__' % pipe_type).default
    del possible_default[possible_default.index(default)]

    # A classifier which can handle sparse data after the densifier is
    # forbidden for memory issues
    for key in components:
        # TODO regression dataset_properties=None
        if SPARSE in availables[key].get_properties()['input']:
            if 'densifier' in preprocessors:
                while True:
                    try:
                        cs.add_forbidden_clause(
                            ForbiddenAndConjunction(
                                ForbiddenEqualsClause(
                                    cs.get_hyperparameter(
                                        '%s:__choice__' % pipe_type), key),
                                ForbiddenEqualsClause(
                                    cs.get_hyperparameter(
                                        'preprocessor:__choice__'), 'densifier')
                            ))
                        # Success
                        break
                    except ValueError:
                        # Change the default and try again
                        try:
                            default = possible_default.pop()
                        except IndexError:
                            raise ValueError("Cannot find a legal default configuration.")
                        cs.get_hyperparameter('%s:__choice__' % pipe_type).default = default

    # which would take too long
    # Combinations of non-linear models with feature learning:
    for c, f in itertools.product(components_, feature_learning_):
        if c not in components:
            continue
        if f not in preprocessors:
            continue
        while True:
            try:
                cs.add_forbidden_clause(ForbiddenAndConjunction(
                    ForbiddenEqualsClause(cs.get_hyperparameter(
                        "%s:__choice__" % pipe_type), c),
                    ForbiddenEqualsClause(cs.get_hyperparameter(
                        "preprocessor:__choice__"), f)))
                break
            except KeyError:
                break
            except ValueError as e:
                # Change the default and try again
                try:
                    default = possible_default.pop()
                except IndexError:
                    raise ValueError(
                        "Cannot find a legal default configuration.")
                cs.get_hyperparameter('%s:__choice__' % pipe_type).default = default


    if task_type in CLASSIFICATION_TASKS:
        # Won't work
        # Multinomial NB etc don't use with features learning, pca etc
        components_ = ["multinomial_nb"]
        preproc_with_negative_X = ["kitchen_sinks", "pca", "truncatedSVD",
                                   "fast_ica", "kernel_pca", "nystroem_sampler"]

        for c, f in itertools.product(components_, preproc_with_negative_X):
            if c not in components:
                continue
            if f not in preprocessors:
                continue
            while True:
                try:
                    cs.add_forbidden_clause(ForbiddenAndConjunction(
                        ForbiddenEqualsClause(cs.get_hyperparameter(
                            "preprocessor:__choice__"), f),
                        ForbiddenEqualsClause(cs.get_hyperparameter(
                            "classifier:__choice__"), c)))
                    break
                except KeyError:
                    break
                except ValueError:
                    # Change the default and try again
                    try:
                        default = possible_default.pop()
                    except IndexError:
                        raise ValueError(
                            "Cannot find a legal default configuration.")
                    cs.get_hyperparameter('classifier:__choice__').default = default

    return cs