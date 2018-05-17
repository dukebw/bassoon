"""Client library code for bassoon."""
import io
import threading

import numpy as np
import torch
import twisted.python.log
import twisted.internet.reactor
import twisted.web.client
import twisted.web.http_headers


def optim_state_to_wire(optim, get_state_fn):
    """Convert data stored in the optimizer `optim` into a numpy vector "wire"
    format that can be transferred over, e.g., TCP.

    Args:
        optim: The `torch.optim.Optimizer` object whose state will be
            serialized.
        get_state_fn: A function to map parameters from
            `optim.param_groups['params']` to the desired state to be
            serialized.
            E.g., `get_state_fn` should be `lambda p: p` to serialize
            parameters, and `lambda p: p.grad` to serialize gradients.

    Returns: a numpy array containing a copy of the desired optimizer state, in
        wire format.
    """
    state = []
    for group in optim.param_groups:
        state += [get_state_fn(p).data.view(-1) for p in group['params']]

    return torch.cat(state).numpy()


def wire_to_optim_state(optim, wire_data, get_state_fn):
    """Convert optimizer state from wire format, and overwrite the current
    state in `optim` with the state sent on the wire (e.g. over TCP).

    Args:
        optim: The `torch.optim.Optimizer` object whose state will be
            overwritten with the state corresponding to the wire data.
        wire_data: A numpy array of data in wire format.
        get_state_fn: A mapping from parameters to optimizer state (see the
            `optim_state_to_wire` docstring).
    """
    start = 0
    for group in optim.param_groups:
        for p in group['params']:
            state = get_state_fn(p)
            if state is None:
                p.grad = torch.autograd.Variable(torch.zeros(p.data.shape))
                state = p.grad

            len_state = len(state.data.view(-1))
            end = start + len_state
            wire_state = torch.FloatTensor(wire_data[start:end])
            state.data[:] = wire_state.view(state.data.size())

            start += len_state


def add_callback(deferred, callback, *args):
    """Add callback to deferred, and set up logging for errors."""
    deferred.addCallback(callback, *args)
    deferred.addErrback(twisted.python.log.err, _why=repr(callback))

    return deferred


def post_body(agent, port, page, body):
    """Posts `body` to /`page` on localhost `port`."""
    uri = f'http://localhost:{port}/{page}'
    return agent.request(method=b'POST',
                         uri=uri.encode('utf-8'),
                         headers=twisted.web.http_headers.Headers(
                             {'User-Agent': ['Test parameter server']}),
                         bodyProducer=body)


def post_data_bytes(agent, data_bytes, port, page):
    """Make a POST request to localhost port with `data_bytes` as the request
    body.
    """
    body = twisted.web.client.FileBodyProducer(
        inputFile=io.BytesIO(data_bytes))

    return post_body(agent, port, page, body)


def _set_buffer(response_bytes, out_buf, sem):
    """Copy the response into out_buf."""
    try:
        out_buf[:] = np.frombuffer(response_bytes, dtype=out_buf.dtype)
    except ValueError:
        print(f'length or value error: {response_bytes}')
        raise

    sem.release()


def _handle_response_recv_buf_cb(response, out_buf, sem):
    """Read the POST response."""
    deferred = twisted.web.client.readBody(response)

    return add_callback(deferred, _set_buffer, out_buf, sem)


def receive_buffer(out_buf, agent, request_content, uri, sem):
    """POSTS to uri and receives a uint8 buffer response.

    Args:
        out_buf: Output buffer.
        agent: Web client.
        request_content: Bytes or numpy array containing content to POST to
            uri.
        uri: uri (port:page) to POST to and receive buffer from.
        sem: Semaphore to release upon receiving buffer.

    Returns: a callback that fills out_buf with the POST response body.

    IMPORTANT(brendan): sem increments upon completion, and should be
    acquired _before_ calling receive_buffer.
    """
    port, page = uri.split(':')

    if not isinstance(request_content, bytes):
        request_content = request_content.tobytes()

    deferred = post_data_bytes(agent, request_content, port, page)

    return add_callback(deferred, _handle_response_recv_buf_cb, out_buf, sem)


def start_reactor():
    """Starts a daemon running twisted.internet.reactor.

    Returns: (agent, semaphore) tuple, where agent is a web client for the
        reactor, and semaphore is a one-valued semaphore.
    """
    t = threading.Thread(target=twisted.internet.reactor.run, args=(False,))
    t.daemon = True
    t.start()

    pool = twisted.web.client.HTTPConnectionPool(
        reactor=twisted.internet.reactor, persistent=True)
    agent = twisted.web.client.Agent(reactor=twisted.internet.reactor,
                                     pool=pool)

    return agent, threading.Semaphore(1)
