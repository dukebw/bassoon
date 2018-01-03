"""A parameter server meant to be per-experiment, which receives an initial set
of parameters, then applies updates to those parameters, returning the updated
parameters.

There should be two functions exposed:

initialize(): takes a set of parameter values as the initial parameters.

reset(): reset the server to its uninitialized state.

update(): updates the stored set of parameters atomically, and returns the
updated parameters to the client.
"""
import numpy as np
import twisted.web.resource
import twisted.web.server
import twisted.internet.endpoints
import twisted.internet.reactor

from . import types


HTTP_SUCCESS_ALREADY_INIT = 299
HTTP_CLIENT_ERR_NOT_INIT = 499


class Params(types.Struct):
    """Wrapper to dependency-inject parameters into the different
    `twisted.web.resource.Resource` implementations.
    """
    _fields = ['val']


def _decode_params(request):
    """Converts parameters serialized as bytes in `request` to a numpy array.
    """
    params = request.content.read()
    return np.fromstring(params, dtype=np.float32)


class Initialize(twisted.web.resource.Resource):
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


class Reset(twisted.web.resource.Resource):
    isLeaf = True

    def __init__(self, params):
        twisted.web.resource.Resource.__init__(self)
        self.params = params

    def render_POST(self, request):
        self.params.val = None

        return bytes()


class Update(twisted.web.resource.Resource):
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

    root = twisted.web.resource.Resource()
    root.putChild(path=b'init', child=Initialize(params))
    root.putChild(path=b'reset', child=Reset(params))
    root.putChild(path=b'update', child=Update(params))

    site = twisted.web.server.Site(resource=root)

    endpoint = twisted.internet.endpoints.TCP4ServerEndpoint(
        reactor=twisted.internet.reactor, port=8880)
    endpoint.listen(site)

    twisted.internet.reactor.run()
