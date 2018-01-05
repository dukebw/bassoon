"""Tests the parameter server.

The test currently is a unit test for the normal use case, when there is a
single process running for each CPU core, all making updates to the parameter
server at once.

TODO(brendan): More black-box tests for the parameter server:

    1. (Performance) test for <= 100ms RTT latency on 1MB transfers.

    2. (Stress) Run 1024 processes all initializing and then iteratively
       updating the parameters.

    3. (Server killed) Test handling for server killed and restarting. How many
       updates are lost? Clients alive or dead?
"""
import collections
import multiprocessing
import threading

import click
import numpy as np
import twisted.internet.defer
import twisted.internet.endpoints
import twisted.internet.protocol
import twisted.internet.reactor
import twisted.web.client

import bassoon.client
import bassoon.parameter_server


# NOTE(brendan): Params contains a value, and a semaphore to enforce sequential
# sending of parameter updates (in order to simulate the normal use case).
Params = collections.namedtuple(typename='Params', field_names='val, sem')


def _post_params(agent, params_val, page):
    """Convenience function for sending parameters to the parameter server as
    bytes.

    Args:
        agent: HTTP client.
        params_val: numpy array of parameter values of type `np.float32`.
        page: Page of URI to POST to.

    Returns the deferred corresponding to completion of the POST request.
    """
    return bassoon.client.post_data_bytes(agent, params_val.tobytes(), page)


def _reset_params(response_param_bytes, params):
    """Resets `params` with the value stored in `response_param_bytes`."""
    params.sem.release()
    returned_params = np.fromstring(response_param_bytes, dtype=np.float32)
    params.val[:] = returned_params.reshape(params.val.shape)


def _handle_response_init_cb(response, params):
    """Reads `response` and initializes `params` with the server's returned
    parameters iff the server has already been initialized.
    """
    if response.code != bassoon.parameter_server.HTTP_SUCCESS_ALREADY_INIT:
        params.sem.release()
        return

    # NOTE(brendan): if the parameters have already been initialized, then the
    # server should return the initial parameters to update the client.
    assert response.length == 4*params.val.shape[0]*params.val.shape[1]

    deferred = twisted.web.client.readBody(response)

    return bassoon.client.add_callback(deferred, _reset_params, params)


def _init_params(agent, params):
    """Initializes the server's parameters by making a POST to /init.

    If the server's parameters are already initialized, the returned parameters
    are filled into `params`.

    Args:
        agent: HTTP client.
        params: Parameters to initialize the server with. `params` will be
            overwritten with the return value from the server if the server's
            parameters have already been initialized.
    """
    deferred = _post_params(agent, params.val, 'init/test')

    return bassoon.client.add_callback(deferred,
                                       _handle_response_init_cb,
                                       params)


def _reset_params_update_cb(response_param_bytes,
                            params,
                            update_counter,
                            num_updates):
    """Reset `params`. See `_handle_response_update_cb` for arguments."""
    _reset_params(response_param_bytes, params)

    with update_counter.get_lock():
        update_counter.value += 1

    if update_counter.value >= num_updates:
        twisted.internet.reactor.stop()


def _handle_response_update_cb(response, params, update_counter, num_updates):
    """Reads the response body of an /update POST, and fills `params` with the
    resulting response.

    Args:
        response: Response returned from the /update POST.

        For other arguments see `_update_params`.

    Returns: A deferred that resets `params`.
    """
    deferred = twisted.web.client.readBody(response)

    return bassoon.client.add_callback(deferred,
                                       _reset_params_update_cb,
                                       params,
                                       update_counter,
                                       num_updates)


def _update_params(params,
                   agent,
                   param_update,
                   update_counter,
                   num_updates):
    """Updates the server's parameters via a POST to /update and a chain of
    deferreds.

    The server returns the updated parameters, which are stored in `params`. In
    this way, parameters are synchronized between client and server.

    Args:
        params: Local worker copy of the parameters, to be filled in.
        agent: HTTP client.
        param_update: Update to make on the parameters.
        update_counter: Counter for number of updates made, so that the reactor
            can be stopped after `num_updates` updates.
        num_updates: Total number of parameter updates to make.

    Returns:
        The deferred chain achieving the parameter update and synchronization.
    """
    deferred = _post_params(agent, param_update, 'update/test')

    return bassoon.client.add_callback(deferred,
                                       _handle_response_update_cb,
                                       params,
                                       update_counter,
                                       num_updates)


def _test_param_server_single_proc(num_updates, params_shape):
    """Makes `num_updates` sequential parameter updates to the server's
    parameters, which are initialized to zero.

    Args:
        num_updates: Number of parameter updates this process should make.
    """
    t = threading.Thread(target=twisted.internet.reactor.run, args=(False,))
    t.start()

    pool = twisted.web.client.HTTPConnectionPool(
        reactor=twisted.internet.reactor, persistent=True)
    agent = twisted.web.client.Agent(reactor=twisted.internet.reactor,
                                     pool=pool)

    update_counter = multiprocessing.Value(typecode_or_type='I')
    update_counter.value = 0

    params = Params(val=np.zeros(params_shape, dtype=np.float32),
                    sem=threading.Semaphore(value=1))

    params.sem.acquire()
    twisted.internet.reactor.callFromThread(_init_params, agent, params)

    param_update = np.ones(params.val.shape, dtype=np.float32)
    for _ in range(num_updates):
        params.sem.acquire()
        twisted.internet.reactor.callFromThread(_update_params,
                                                params,
                                                agent,
                                                param_update,
                                                update_counter,
                                                num_updates)

    t.join()


def _check_updates_applied_cb(response_bytes, num_updates):
    """Checks that all updates were applied based on `response_bytes`, which
    contains the parameters stored on the server, by checking that
    nproc*num_updates is the value of each element of the parameters.

    Args:
        same as `_read_check_response_cb`.
    """
    twisted.internet.reactor.stop()

    server_params = np.fromstring(response_bytes, dtype=np.float32)

    num_proc = multiprocessing.cpu_count()
    assert np.all(server_params == float(num_proc*num_updates))


def _read_check_response_cb(response, num_updates):
    """Reads a response body from a POST to /init, and checks that all updates
    were applied.

    Args:
        response_bytes: Bytes of the response returned from the /init POST to
            the server.
        num_updates: Number of parameter updates to make.
    """
    deferred = twisted.web.client.readBody(response)
    return bassoon.client.add_callback(deferred,
                                       _check_updates_applied_cb,
                                       num_updates)


def _check_test_results(agent, params, num_updates):
    """To get the current parameters stored on the server, a POST to /init
    is made.

    Args:
        agent: HTTP client.
        params: Dummy parameters to send to the parameter server, just to get
            the already-initialized response.
    """
    deferred = _post_params(agent, params, 'init/test')
    return bassoon.client.add_callback(deferred,
                                       _read_check_response_cb,
                                       num_updates)


def _reset_param_server_cb(agent):
    """Callback function to POST a to /reset, and stop the reactor
    afterwards.
    """
    deferred = bassoon.client.post_body(agent, 'reset/test', None)

    return bassoon.client.add_callback(
        deferred, lambda x: twisted.internet.reactor.stop())


def _reset_param_server_params():
    """Makes a call to reset the parameter server's parameters."""
    t = threading.Thread(target=twisted.internet.reactor.run, args=(False,))
    t.start()

    agent = twisted.web.client.Agent(reactor=twisted.internet.reactor)

    twisted.internet.reactor.callFromThread(_reset_param_server_cb, agent)

    t.join()


@click.command()
@click.option('--num-updates',
              default=32,
              help='Number of parameter updates to test normal path.')
@click.option('--params-shape', default=(1024, 1024), nargs=2, type=int)
def test_param_server(num_updates, params_shape):
    """Test the parameter server's normal path.

    `nproc` processes are launched, all of which make `num_updates` updates to
    the parameter server. The parameters start at zero, and each update is an
    array of all ones. So, the test can check that all updates went through by
    checking that the final parameters contain nproc*num_updates in all
    elements.
    """
    # NOTE(brendan): The server's parameters are reset from a separate process,
    # in order to prevent any issues with `twisted.internet.reactor` already
    # running when the `_test_param_server_single_proc` processes are forked.
    p = multiprocessing.Process(target=_reset_param_server_params)
    p.start()
    p.join()

    processes = []
    for _ in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(target=_test_param_server_single_proc,
                                    args=(num_updates, params_shape))
        p.start()

        processes.append(p)

    for p in processes:
        p.join()

    t = threading.Thread(target=twisted.internet.reactor.run, args=(False,))
    t.start()

    agent = twisted.web.client.Agent(reactor=twisted.internet.reactor)

    params = np.zeros(params_shape, dtype=np.float32)
    twisted.internet.reactor.callFromThread(_check_test_results,
                                            agent,
                                            params,
                                            num_updates)

    t.join()

    print('Test passed!')
