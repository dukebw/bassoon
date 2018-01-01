"""Tests the parameter server."""
import io
import multiprocessing
import threading
import time

import click
import numpy as np
import twisted.internet.defer
import twisted.internet.endpoints
import twisted.internet.protocol
import twisted.internet.reactor
import twisted.web.client

import bassoon.parameter_server


PARAMS_SHAPE = [1024, 1024]


def _reset_params(response_param_bytes, params):
    """
    """
    params[:] = np.fromstring(response_param_bytes.decode('utf-8'),
                              dtype=np.float32)


def _handle_response_init(response, start_time, param_sem, params):
    """
    """
    print('Elapsed time: {}'.format(time.time() - start_time))
    param_sem.release()

    if response.code != bassoon.parameter_server.HTTP_STATUS_ALREADY_INIT:
        return

    # NOTE(brendan): if the parameters have already been initialized, then the
    # server should return the initial parameters to update the client.
    assert response.length == 4*PARAMS_SHAPE[0]*PARAMS_SHAPE[1]

    deferred = twisted.web.client.readBody(response)
    deferred.addCallback(_reset_params, params)

    return deferred


def _handle_response_update(response,
                            start_time,
                            param_sem,
                            update_counter,
                            num_updates):
    """
    """
    print('Elapsed time: {}'.format(time.time() - start_time))
    param_sem.release()

    with update_counter.get_lock():
        update_counter.value += 1

    if update_counter.value >= num_updates:
        twisted.internet.reactor.stop()


def _post_params(agent, params, page):
    """Make a POST request to localhost port 8880 with `params` as the request
    body.
    """
    body = twisted.web.client.FileBodyProducer(
        inputFile=io.BytesIO(params.tobytes()))

    uri = 'http://localhost:8880/{}'.format(page)
    return agent.request(method=b'POST',
                         uri=uri.encode('utf-8'),
                         headers=twisted.web.http_headers.Headers(),
                         bodyProducer=body)


def _init_params(agent, param_sem, params):
    """
    """
    start_time = time.time()

    deferred = _post_params(agent, params, 'init')

    deferred.addCallback(_handle_response_init, start_time, param_sem, params)

    return deferred


def _update_params(agent, param_sem, param_update, update_counter, num_updates):
    """
    """
    start_time = time.time()

    deferred = _post_params(agent, param_update, 'update')

    deferred.addCallback(_handle_response_update,
                         start_time,
                         param_sem,
                         update_counter,
                         num_updates)

    return deferred


def _test_param_server_single_proc(num_updates):
    """
    """
    t = threading.Thread(target=twisted.internet.reactor.run, args=(False,))
    t.start()

    param_sem = threading.Semaphore(value=1)

    pool = twisted.web.client.HTTPConnectionPool(
        reactor=twisted.internet.reactor, persistent=True)
    agent = twisted.web.client.Agent(reactor=twisted.internet.reactor,
                                     pool=pool)

    update_counter = multiprocessing.Value(typecode_or_type='I')
    params = np.zeros(PARAMS_SHAPE, dtype=np.float32)
    twisted.internet.reactor.callFromThread(_init_params,
                                            agent,
                                            param_sem,
                                            params)

    param_update = np.ones(PARAMS_SHAPE, dtype=np.float32)
    for _ in range(num_updates):
        param_sem.acquire()
        twisted.internet.reactor.callFromThread(_update_params,
                                                agent,
                                                param_sem,
                                                param_update,
                                                update_counter,
                                                num_updates)

    t.join()


@click.command()
@click.option('--num-updates',
              default=32,
              help='Number of parameter updates to test normal path.')
def test_param_server(num_updates):
    """Test the parameter server."""
    processes = []
    for _ in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(target=_test_param_server_single_proc,
                                    args=(num_updates,))
        p.start()

        processes.append(p)

    for p in processes:
        p.join()

    t = threading.Thread(target=twisted.internet.reactor.run, args=(False,))
    t.start()

    agent = twisted.web.client.Agent(reactor=twisted.internet.reactor)

    params = np.zeros(PARAMS_SHAPE, dtype=np.float32)
    # TODO(brendan): Clean this up and assert that the parameters have been
    # updated correctly, according to the number of updates.
    def _print_response(r):
        deferred = twisted.web.client.readBody(r)
        deferred.addCallback(lambda r_bytes: print(np.fromstring(r_bytes.decode('utf-8'), dtype=np.uint8).shape))

        return deferred

    def _print_params():
        deferred = _post_params(agent, params, 'init')
        deferred.addCallback(_print_response)

        return deferred

    twisted.internet.reactor.callFromThread(_print_params)

    t.join()
