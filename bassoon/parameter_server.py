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

    def render_POST(self, request):  # pylint:disable=unused-argument
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


def _gen_random_arch(num_nodes, rank):
    """Generates one uniformly random architecture with num_nodes nodes."""
    binary_ops = np.random.binomial(n=1, p=0.5, size=num_nodes)

    act_fns = np.random.multinomial(n=1,
                                    pvals=4*[0.25],
                                    size=2*num_nodes)
    # NOTE(brendan): Converts from one-hot vector to index.
    act_fns = np.argmax(act_fns, axis=1)

    in_nodes = [np.random.multinomial(n=1, pvals=i*[1/i], size=2)
                for i in range(rank, rank + num_nodes)]
    in_nodes = [np.argmax(n, axis=1) for n in in_nodes]
    in_nodes = np.concatenate(in_nodes)

    # NOTE(brendan): wire_arch is of length
    # num_nodes + 2*num_nodes + 2*num_nodes
    wire_arch = np.concatenate([binary_ops, act_fns, in_nodes])

    return wire_arch.astype(np.uint8)


class FusionSharedTrainDriver(twisted.web.resource.Resource):
    """Returns an epoch of architectures on POST."""
    isLeaf = True

    def __init__(self):
        twisted.web.resource.Resource.__init__(self)

    def render_POST(self, request):
        """Generates a uniformly random epoch of architectures with num_nodes
        nodes.
        """
        request_content = request.content.read()
        request_content = np.frombuffer(request_content, dtype=np.int32)

        minibatches_per_epoch = request_content[0]
        num_nodes = request_content[1]
        rank = request_content[2]

        epoch_archs = []
        for _ in range(minibatches_per_epoch):
            wire_arch = _gen_random_arch(num_nodes, rank)

            epoch_archs.append(wire_arch)

        epoch_archs = np.concatenate(epoch_archs).tobytes()

        return epoch_archs


class FusionSharedValDriverArch(twisted.web.resource.Resource):
    isLeaf = True

    def __init__(self):
        twisted.web.resource.Resource.__init__(self)

    def render_POST(self, request):
        """Returns one uniformly random architecture."""
        request_content = request.content.read()
        request_content = np.frombuffer(request_content, dtype=np.int32)

        num_nodes = request_content[0]
        rank = request_content[1]
        arch = _gen_random_arch(num_nodes, rank)

        return arch.tobytes()


class FusionSharedValDriverReward(twisted.web.resource.Resource):
    isLeaf = True

    def __init__(self):
        twisted.web.resource.Resource.__init__(self)

    def render_POST(self, request):  # pylint:disable=unused-argument
        """ACKs."""
        return bytes()


class FusionControllerTrainStub(twisted.web.resource.Resource):
    isLeaf = True

    def __init__(self):
        twisted.web.resource.Resource.__init__(self)

    def render_POST(self, request):  # pylint:disable=unused-argument
        """Return random reward."""
        reward = np.clip(
            np.random.normal(loc=0.5, scale=0.1), a_min=0.0, a_max=1.0)
        return np.float32(reward).tobytes()


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

    def render_POST(self, request):  # pylint:disable=unused-argument
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

    fusion = twisted.web.resource.Resource()
    root.putChild(path=b'fusion', child=fusion)

    fusion_shared_train_driver = FusionSharedTrainDriver()
    fusion.putChild(path=b'shared-train-driver',
                    child=fusion_shared_train_driver)

    fusion_shared_val_driver = twisted.web.resource.Resource()
    fusion.putChild(path=b'shared-val-driver',
                    child=fusion_shared_val_driver)

    fusion_shared_val_arch = FusionSharedValDriverArch()
    fusion_shared_val_driver.putChild(path=b'arch',
                                      child=fusion_shared_val_arch)

    fusion_shared_val_reward = FusionSharedValDriverReward()
    fusion_shared_val_driver.putChild(path=b'reward',
                                      child=fusion_shared_val_reward)

    fusion_controller_train_stub = FusionControllerTrainStub()
    fusion.putChild(path=b'controller-train-stub',
                    child=fusion_controller_train_stub)

    site = twisted.web.server.Site(resource=root)

    endpoint = twisted.internet.endpoints.TCP4ServerEndpoint(
        reactor=twisted.internet.reactor, port=8880)
    endpoint.listen(site)

    twisted.internet.reactor.run()
