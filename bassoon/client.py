"""Client library code for bassoon."""
import io

import torch
import twisted


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
            state.data[:] = torch.FloatTensor(wire_data[start:end])

            start += len_state


def add_callback(deferred, callback, *args):
    """Add callback to deferred, and set up logging for errors."""
    deferred.addCallback(callback, *args)
    deferred.addErrback(twisted.python.log.err, _why=repr(callback))

    return deferred


def post_body(agent, page, body):
    """Posts `body` to /`page` on localhost port 8880."""
    uri = 'http://localhost:8880/{}'.format(page)
    return agent.request(method=b'POST',
                         uri=uri.encode('utf-8'),
                         headers=twisted.web.http_headers.Headers(
                             {'User-Agent': ['Test parameter server']}),
                         bodyProducer=body)


def post_data_bytes(agent, data_bytes, page):
    """Make a POST request to localhost port 8880 with `data_bytes` as the
    request body.
    """
    body = twisted.web.client.FileBodyProducer(
        inputFile=io.BytesIO(data_bytes))

    return post_body(agent, page, body)
