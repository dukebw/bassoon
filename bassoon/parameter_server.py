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

import click
import numpy as np
import twisted.web.resource
import twisted.web.server
import twisted.internet.defer
import twisted.internet.endpoints
import twisted.internet.reactor

from . import types
from . import client


HTTP_SUCCESS_ALREADY_INIT = 299
HTTP_CLIENT_ERR_NOT_INIT = 499
HTTP_CLIENT_ERR_INVALID_PARAMS = 498

NODE_SIZE = 5


class Injection(types.Struct):
    """Wrapper to dependency-inject stuff into the different
    `twisted.web.resource.Resource` implementations.
    """
    _fields = ['val']


class SemaphoreInjection(types.Struct):
    """Dependency injection of a resource that requires a semaphore."""
    _fields = ['val', 'sem']


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
    # TODO(brendan): This is no longer correct: New format is
    # [binary_op, act_fn1, act_fn2, node1, node2] since this is output from the
    # controller.
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


def _read_shared_train(request):
    """Read and return request content from shared model trainer."""
    request_content = request.content.read()
    request_content = np.frombuffer(request_content, dtype=np.int32)

    minibatches_per_epoch = request_content[0]
    num_nodes = request_content[1]
    rank = request_content[2]

    return minibatches_per_epoch, num_nodes, rank


def _epoch_archs_response_cb(deferred, request, epoch_archs, requested_bytes):
    """Respond with epoch_archs bytes."""
    if requested_bytes != len(epoch_archs.val):
        request.setResponseCode(HTTP_CLIENT_ERR_INVALID_PARAMS)

        request.write(bytes())
        request.finish()

    request.write(epoch_archs.val.tobytes())
    request.finish()


class FusionSharedTrain(twisted.web.resource.Resource):
    """Respond to request for an epoch of architectures."""

    isLeaf = True

    def __init__(self, epoch_archs, minibatches_per_epoch):
        twisted.web.resource.Resource.__init__(self)
        self.epoch_archs = epoch_archs
        self.minibatches_per_epoch = minibatches_per_epoch

    def render_POST(self, request):
        """Return an epoch of architectures once available."""
        minibatches_per_epoch, num_nodes, rank = _read_shared_train(request)

        if self.minibatches_per_epoch.val is None:
            self.minibatches_per_epoch.val = minibatches_per_epoch
            self.minibatches_per_epoch.sem.release()

        deferred = self.epoch_archs.sem.acquire()

        client.add_callback(deferred,
                            _epoch_archs_response_cb,
                            request,
                            self.epoch_archs,
                            minibatches_per_epoch*num_nodes*rank)

        return twisted.web.server.NOT_DONE_YET


def _read_shared_val_arch(request):
    """Read arch request content from validate_rl script."""
    request_content = request.content.read()
    request_content = np.frombuffer(request_content, dtype=np.int32)

    num_nodes = request_content[0]
    rank = request_content[1]

    return num_nodes, rank


def _val_arch_response_cb(deferred, request, val_arch):
    """Respond with a single architecture for validation."""
    num_nodes, _ = _read_shared_val_arch(request)

    val_arch = val_arch.val
    if NODE_SIZE*num_nodes != len(val_arch):
        request.setResponseCode(HTTP_CLIENT_ERR_INVALID_PARAMS)

        request.write(bytes())
        request.finish()

    request.write(val_arch.tobytes())
    request.finish()


class FusionSharedValArch(twisted.web.resource.Resource):
    """Send an available architecture to validate_rl script."""

    isLeaf = True

    def __init__(self, val_arch):
        twisted.web.resource.Resource.__init__(self)
        self.val_arch = val_arch

    def render_POST(self, request):
        """Wait on validation architecture then send it."""
        deferred = self.val_arch.sem.acquire()

        client.add_callback(deferred,
                            _val_arch_response_cb,
                            request,
                            self.val_arch)

        return twisted.web.server.NOT_DONE_YET


class FusionSharedValReward(twisted.web.resource.Resource):
    """Receive a reward."""

    isLeaf = True

    def __init__(self, val_reward):
        twisted.web.resource.Resource.__init__(self)
        self.val_reward = val_reward

    def render_POST(self, request):
        """Receive a validation minibatch's reward and increment semaphore."""
        request_content = request.content.read()
        reward = np.frombuffer(request_content, dtype=np.float32)

        self.val_reward.val = np.frombuffer(reward, dtype=np.float32)
        self.val_reward.sem.release()

        return bytes()


def _controller_step_response_cb(deferred, request, val_reward):
    """Respond with validation reward."""
    request.write(val_reward.val.tobytes())
    request.finish()


class FusionControllerStep(twisted.web.resource.Resource):
    """Handle fusion controller step."""

    isLeaf = True

    def __init__(self, val_arch, val_reward):
        twisted.web.resource.Resource.__init__(self)
        self.val_arch = val_arch
        self.val_reward = val_reward

    def render_POST(self, request):
        """Read a validation architecture, signal its semaphore, then wait for
        the reward.
        """
        request_content = request.content.read()
        arch = np.frombuffer(request_content, dtype=np.uint8)

        self.val_arch.val = arch
        self.val_arch.sem.release()

        deferred = self.val_reward.sem.acquire()
        client.add_callback(deferred,
                            _controller_step_response_cb,
                            request,
                            self.val_reward)

        return twisted.web.server.NOT_DONE_YET


class FusionControllerEpochArch(twisted.web.resource.Resource):
    """Epoch architecture receiver."""

    isLeaf = True

    def __init__(self, epoch_archs):
        twisted.web.resource.Resource.__init__(self)
        self.epoch_archs = epoch_archs

    def render_POST(self, request):
        """Receive an epoch of architectures and signal."""
        request_content = request.content.read()
        epoch_archs = np.frombuffer(request_content, dtype=np.uint8)

        self.epoch_archs.val = epoch_archs
        self.epoch_archs.sem.release()

        return bytes()


def _minibatches_response_cb(deferred, request, minibatches_per_epoch):
    """Respond with minibatches per epoch."""
    request.write(minibatches_per_epoch.val.tobytes())
    request.finish()


class FusionControllerMinibatches(twisted.web.resource.Resource):
    """Transfer minibatches per epoch information."""

    isLeaf = True

    def __init__(self, minibatches_per_epoch):
        twisted.web.resource.Resource.__init__(self)
        self.minibatches_per_epoch = minibatches_per_epoch

    def render_POST(self, request):  # pylint:disable=unused-argument
        """Send minibatches per epoch once available."""
        if self.minibatches_per_epoch.val is None:
            deferred = self.minibatches_per_epoch.sem.acquire()

            client.add_callback(deferred,
                                _minibatches_response_cb,
                                request,
                                self.minibatches_per_epoch)

            return twisted.web.server.NOT_DONE_YET

        return self.minibatches_per_epoch.val.tobytes()


class FusionSharedTrainDriver(twisted.web.resource.Resource):
    """Return an epoch of architectures on POST."""

    isLeaf = True

    def __init__(self):
        twisted.web.resource.Resource.__init__(self)

    def render_POST(self, request):
        """Generates a uniformly random epoch of architectures with num_nodes
        nodes.
        """
        minibatches_per_epoch, num_nodes, rank = _read_shared_train(request)

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
        num_nodes, rank = _read_shared_val_arch(request)
        arch = _gen_random_arch(num_nodes, rank)

        return arch.tobytes()


class AckStub(twisted.web.resource.Resource):
    """Stub that always acks."""

    isLeaf = True

    def __init__(self):
        twisted.web.resource.Resource.__init__(self)

    def render_POST(self, request):  # pylint:disable=unused-argument
        """ACKs."""
        return bytes()


class FusionControllerStepStub(twisted.web.resource.Resource):
    isLeaf = True

    def __init__(self):
        twisted.web.resource.Resource.__init__(self)

    def render_POST(self, request):  # pylint:disable=unused-argument
        """Return random reward."""
        reward = np.clip(
            np.random.normal(loc=0.5, scale=0.1), a_min=0.0, a_max=1.0)
        return np.float32(reward).tobytes()


class FusionControllerMinibatchesStub(twisted.web.resource.Resource):
    """Stub to return minibatches per epoch."""

    isLeaf = True

    def __init__(self):
        twisted.web.resource.Resource.__init__(self)

    def render_POST(self, request):  # pylint:disable=unused-argument
        """Return number of minibatches per epoch."""
        return np.int32(1500).tobytes()


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


def _get_semaphore_injection():
    """Create a semaphore initialized to zero.

    Return a DeferredSemaphore with token limit of 1.
    """
    # NOTE(brendan): DeferredSemaphore requires initialization with tokens > 1.
    semaphore_injection = SemaphoreInjection(
        val=None, sem=twisted.internet.defer.DeferredSemaphore(1))

    semaphore_injection.sem.tokens = 0

    return semaphore_injection


@click.command()
@click.option('--port', default=None, help='Port to listen on.')
def parameter_server(port):
    """Runs a parameter server that stores and updates a set of parameters."""
    params = Injection(val=None)
    optim = Injection(val=None)

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

    epoch_archs = _get_semaphore_injection()
    minibatches_per_epoch = _get_semaphore_injection()
    fusion_shared_train = FusionSharedTrain(epoch_archs, minibatches_per_epoch)
    fusion.putChild(path=b'shared-train', child=fusion_shared_train)

    fusion_shared_val = twisted.web.resource.Resource()
    fusion.putChild(path=b'shared-val', child=fusion_shared_val)

    val_arch = _get_semaphore_injection()
    fusion_shared_val_arch = FusionSharedValArch(val_arch)
    fusion_shared_val.putChild(path=b'arch', child=fusion_shared_val_arch)

    val_reward = _get_semaphore_injection()
    fusion_shared_val_reward = FusionSharedValReward(val_reward)
    fusion_shared_val.putChild(path=b'reward', child=fusion_shared_val_reward)

    controller_train = twisted.web.resource.Resource()
    fusion.putChild(path=b'controller-train', child=controller_train)

    fusion_controller_step = FusionControllerStep(val_arch, val_reward)
    controller_train.putChild(path=b'step', child=fusion_controller_step)

    fusion_controller_epoch_arch = FusionControllerEpochArch(epoch_archs)
    controller_train.putChild(path=b'epoch_arch',
                              child=fusion_controller_epoch_arch)

    fusion_controller_minibatches = FusionControllerMinibatches(
        minibatches_per_epoch)
    controller_train.putChild(path=b'minibatches_per_epoch',
                              child=fusion_controller_minibatches)

    # NOTE(brendan): Drivers and stubs for development
    fusion_shared_train_driver = FusionSharedTrainDriver()
    fusion.putChild(path=b'shared-train-driver',
                    child=fusion_shared_train_driver)

    fusion_shared_val_driver = twisted.web.resource.Resource()
    fusion.putChild(path=b'shared-val-driver',
                    child=fusion_shared_val_driver)

    fusion_shared_val_driver_arch = FusionSharedValDriverArch()
    fusion_shared_val_driver.putChild(path=b'arch',
                                      child=fusion_shared_val_driver_arch)

    fusion_shared_val_driver_reward = AckStub()
    fusion_shared_val_driver.putChild(path=b'reward',
                                      child=fusion_shared_val_driver_reward)

    fusion_controller_train_stub = twisted.web.resource.Resource()
    fusion.putChild(path=b'controller-train-stub',
                    child=fusion_controller_train_stub)

    fusion_controller_step_stub = FusionControllerStepStub()
    fusion_controller_train_stub.putChild(path=b'step',
                                          child=fusion_controller_step_stub)

    fusion_controller_minibatches = FusionControllerMinibatchesStub()
    fusion_controller_train_stub.putChild(path=b'minibatches_per_epoch',
                                          child=fusion_controller_minibatches)

    fusion_controller_epoch_arch = AckStub()
    fusion_controller_train_stub.putChild(path=b'epoch_arch',
                                          child=fusion_controller_epoch_arch)

    site = twisted.web.server.Site(resource=root)

    endpoint = twisted.internet.endpoints.TCP4ServerEndpoint(
        reactor=twisted.internet.reactor, port=port)
    endpoint.listen(site)

    twisted.internet.reactor.run()
