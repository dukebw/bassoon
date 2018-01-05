"""A parameter server meant to be per-experiment, which receives an initial set
of parameters and optimization state, then applies updates to those parameters,
returning the updated parameters.

There are three exposed functions:

initialize(): takes a set of parameter values as the initial parameters.

reset(): reset the server to its uninitialized state.

update(): updates the stored set of parameters atomically, and returns the
updated parameters to the client.

The above functions control the following state machine:

        POST
        RESET
   +-------------+
   |             |
   |     POST    |
   v     INIT    +
UNINIT +------> INIT<--+POST
                   +   |UPDATE
                   +---+

Where UNINIT is the server's state on startup, INIT is the server's state after
calling initialize(), and POST INIT, POST UPDATE and POST RESET correspond to
initialize(), update() and reset(), respectively.

See the `render_POST` method docstrings for the `twisted.web.resource.Resource`
subclasses in this module for detailed descriptions.
"""
import pickle

import numpy as np
import twisted.web.resource
import twisted.web.server
import twisted.internet.endpoints
import twisted.internet.reactor

from . import types
from . import client


HTTP_SUCCESS_ALREADY_INIT = 299
HTTP_CLIENT_ERR_NOT_INIT = 499


class Params(types.Struct):
    """Wrapper to dependency-inject parameters into the different
    `twisted.web.resource.Resource` implementations.
    """
    _fields = ['val']


class Optim(types.Struct):
    """Wrapper to dependency-inject a PyTorch optimizer."""
    _fields = ['val']


class Initialize(twisted.web.resource.Resource):
    """A class handling POST requests to /init."""
    def __init__(self, optim):
        twisted.web.resource.Resource.__init__(self)
        self.optim = optim

    def render_POST(self, request):
        """A POST to /init takes a pickled `torch.optim.Optimizer` as the
        request.

        There are two possible successful reactions to an /init POST:

            1. If the parameter server is in an uninitialized state, the
               optimizer is used to initialize the parameter server and status
               200 is returned with an empty response.

            2. If the parameter server is already initialized, the model
               parameters are returned using the
               `bassoon.client.optim_state_to_wire` API.
        """
        if self.optim.val is None:
            optim_bytes = request.content.read()
            self.optim.val = pickle.loads(optim_bytes)

            return bytes()

        request.setResponseCode(HTTP_SUCCESS_ALREADY_INIT)

        params = client.optim_state_to_wire(self.optim.val, lambda p: p)

        return params.tostring()


class Reset(twisted.web.resource.Resource):
    """A class handling POST requests to /reset."""
    def __init__(self, optim):
        twisted.web.resource.Resource.__init__(self)
        self.optim = optim

    def render_POST(self, request):  # pylint: disable=unused-argument
        """A POST to /reset returns the server to the uninitialized state."""
        self.optim.val = None

        return bytes()


class Update(twisted.web.resource.Resource):
    """A class handling POST requests to /update."""
    def __init__(self, optim):
        twisted.web.resource.Resource.__init__(self)
        self.optim = optim

    def render_POST(self, request):
        """A POST to /update takes a serialized set of gradients corresponding
        to the optimizer that the parameter server has been initialized with.

        Gradients in the request should be serialized using
        `bassoon.client.optim_state_to_wire`.

        The parameter server runs the optimizer for a single step using the
        gradients from the wire. The updated parameters are then returned in
        the wire format from `bassoon.client.optim_state_to_wire`.
        """
        if self.optim.val is None:
            request.setResponseCode(HTTP_CLIENT_ERR_NOT_INIT)

            return bytes()

        grad_bytes = request.content.read()
        grads = np.fromstring(grad_bytes, dtype=np.float32)

        client.wire_to_optim_state(self.optim.val, grads, lambda p: p.grad)

        self.optim.val.step()

        params = client.optim_state_to_wire(self.optim.val, lambda p: p)

        return params.tostring()


def _decode_params(request):
    """Converts parameters serialized as bytes in `request` to a numpy array.
    """
    params = request.content.read()

    return np.fromstring(params, dtype=np.float32)


class InitializeTest(twisted.web.resource.Resource):
    isLeaf = True

    def __init__(self, params):
        twisted.web.resource.Resource.__init__(self)
        self.params = params

    def render_POST(self, request):
        if self.params.val is None:
            self.params.val = _decode_params(request)

            return bytes()

        request.setResponseCode(HTTP_SUCCESS_ALREADY_INIT)

        return self.params.val.tostring()


class ResetTest(twisted.web.resource.Resource):
    isLeaf = True

    def __init__(self, params):
        twisted.web.resource.Resource.__init__(self)
        self.params = params

    def render_POST(self, request):  # pylint: disable=unused-argument
        self.params.val = None

        return bytes()


class UpdateTest(twisted.web.resource.Resource):
    isLeaf = True

    def __init__(self, params):
        twisted.web.resource.Resource.__init__(self)
        self.params = params

    def render_POST(self, request):
        if self.params.val is None:
            request.setResponseCode(HTTP_CLIENT_ERR_NOT_INIT)
            return bytes()

        self.params.val += _decode_params(request)

        return self.params.val.tostring()


def parameter_server():
    """Runs a parameter server that stores and updates a set of parameters."""
    params = Params(val=None)
    optim = Optim(val=None)

    root = twisted.web.resource.Resource()
    init = Initialize(optim)
    reset = Reset(optim)
    update = Update(optim)
    root.putChild(path=b'init', child=init)
    root.putChild(path=b'reset', child=reset)
    root.putChild(path=b'update', child=update)

    init.putChild(path=b'test', child=InitializeTest(params))
    reset.putChild(path=b'test', child=ResetTest(params))
    update.putChild(path=b'test', child=UpdateTest(params))

    site = twisted.web.server.Site(resource=root)

    endpoint = twisted.internet.endpoints.TCP4ServerEndpoint(
        reactor=twisted.internet.reactor, port=8880)
    endpoint.listen(site)

    twisted.internet.reactor.run()
