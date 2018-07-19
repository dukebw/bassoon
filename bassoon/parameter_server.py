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
import sys

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


def _resource_response_cb(deferred, request, resource):
    """Respond with resource value once deferred is signaled."""
    request.write(resource.val.tobytes())
    request.finish()


def _defer_resource_response(resource, request):
    """Defer response until resource is signaled."""
    deferred = resource.sem.acquire()
    client.add_callback(deferred,
                        _resource_response_cb,
                        request,
                        resource)

    return twisted.web.server.NOT_DONE_YET


def _read_shared_train_iter_arch(request):
    """Read and return request content for an iteration architecture from
    shared model trainer.
    """
    request_content = request.content.read()

    num_metadata = 2
    metadata = np.frombuffer(request_content[:num_metadata*4], dtype=np.int32)
    ex_per_minibatch = metadata[0]
    minibatches_per_epoch = metadata[1]

    features = np.frombuffer(request_content[num_metadata*4:],
                             dtype=np.float32)

    return ex_per_minibatch, minibatches_per_epoch, features


class FusionSharedIterArch(twisted.web.resource.Resource):
    """Respond to request for an architecture."""

    isLeaf = True

    def __init__(self, iter_arch, minibatches_per_epoch, train_ex_features):
        twisted.web.resource.Resource.__init__(self)
        self.iter_arch = iter_arch
        self.minibatches_per_epoch = minibatches_per_epoch
        self.train_ex_features = train_ex_features

    def render_POST(self, request):
        """Return an architecture once available."""
        (ex_per_minibatch,
         minibatches_per_epoch,
         features) = _read_shared_train_iter_arch(request)

        if self.minibatches_per_epoch.val is None:
            self.minibatches_per_epoch.val = np.array(
                [ex_per_minibatch, minibatches_per_epoch], dtype=np.int32)
            self.minibatches_per_epoch.sem.release()

        self.train_ex_features.val = features
        self.train_ex_features.sem.release()

        return _defer_resource_response(self.iter_arch, request)


def _read_shared_val_arch(request):
    """Read arch request content from validate_rl script."""
    request_content = request.content.read()
    features = np.frombuffer(request_content, dtype=np.float32)

    return features


class FusionSharedValArch(twisted.web.resource.Resource):
    """Send an available architecture to validate_rl script."""

    isLeaf = True

    def __init__(self, obs, val_arch):
        twisted.web.resource.Resource.__init__(self)

        self.obs = obs
        self.val_arch = val_arch

    def render_POST(self, request):
        """Wait on validation architecture then send it."""
        obs = _read_shared_val_arch(request)

        self.obs.val = obs
        self.obs.sem.release()

        return _defer_resource_response(self.val_arch, request)


def _receive_resource(resource, request, dtype):
    """Read from request into resource, and signal."""
    request_content = request.content.read()

    resource.val = np.frombuffer(request_content, dtype=dtype)
    resource.sem.release()

    return bytes()


class FusionSharedValReward(twisted.web.resource.Resource):
    """Receive a reward."""

    isLeaf = True

    def __init__(self, val_reward):
        twisted.web.resource.Resource.__init__(self)
        self.val_reward = val_reward

    def render_POST(self, request):
        """Receive a validation minibatch's reward and increment semaphore."""
        return _receive_resource(self.val_reward, request, dtype=np.float32)


class FusionControllerStepReward(twisted.web.resource.Resource):
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

        return _defer_resource_response(self.val_reward, request)


class FusionControllerGetTrainFeatures(twisted.web.resource.Resource):
    """Receive fused feature observations."""

    isLeaf = True

    def __init__(self, train_ex_features):
        twisted.web.resource.Resource.__init__(self)
        self.train_ex_features = train_ex_features

    def render_POST(self, request):
        """Wait on and return fused features."""
        return _defer_resource_response(self.train_ex_features, request)


class FusionControllerGetObs(twisted.web.resource.Resource):
    """Receive observations from shared val model."""

    isLeaf = True

    def __init__(self, obs):
        twisted.web.resource.Resource.__init__(self)
        self.obs = obs

    def render_POST(self, request):
        """Wait on and return obs."""
        return _defer_resource_response(self.obs, request)


class FusionControllerIterArch(twisted.web.resource.Resource):
    """Iteration architecture receiver."""

    isLeaf = True

    def __init__(self, iter_arch):
        twisted.web.resource.Resource.__init__(self)
        self.iter_arch = iter_arch

    def render_POST(self, request):
        """Receive an architecture and signal."""
        return _receive_resource(self.iter_arch, request, dtype=np.uint8)


class FusionControllerMinibatches(twisted.web.resource.Resource):
    """Transfer minibatches per epoch information.

    NOTE(brendan): minibatches per epoch is
    [ex_per_minibatch, minibatches_per_epoch].
    """

    isLeaf = True

    def __init__(self, minibatches_per_epoch):
        twisted.web.resource.Resource.__init__(self)
        self.minibatches_per_epoch = minibatches_per_epoch

    def render_POST(self, request):  # pylint:disable=unused-argument
        """Send minibatches per epoch once available."""
        if self.minibatches_per_epoch.val is None:
            return _defer_resource_response(self.minibatches_per_epoch,
                                            request)

        return self.minibatches_per_epoch.val.tobytes()


class FusionSharedSendCheckpoint(twisted.web.resource.Resource):
    """Send checkpoint from train.py."""

    isLeaf = True

    def __init__(self, checkpoint):
        twisted.web.resource.Resource.__init__(self)
        self.checkpoint = checkpoint

    def render_POST(self, request):
        """Set and signal checkpoint."""
        return _receive_resource(self.checkpoint, request, dtype=np.uint8)


class FusionSharedValCheckpoint(twisted.web.resource.Resource):
    """Receive checkpoint from shared train."""

    isLeaf = True

    def __init__(self, checkpoint):
        twisted.web.resource.Resource.__init__(self)
        self.checkpoint = checkpoint

    def render_POST(self, request):  # pylint:disable=unused-argument
        """Wait and receive checkpoint."""
        return _defer_resource_response(self.checkpoint, request)


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
    # NOTE(brendan): DeferredSemaphore requires initialization with
    # tokens >= 1.
    semaphore_injection = SemaphoreInjection(
        val=None, sem=twisted.internet.defer.DeferredSemaphore(1))

    semaphore_injection.sem.tokens = 0

    return semaphore_injection


def _setup_shared_val(fusion, checkpoint):
    """Setup pages for shared model validation, and return arch/reward
    resources.
    """
    fusion_shared_val = twisted.web.resource.Resource()
    fusion.putChild(path=b'shared-val', child=fusion_shared_val)

    obs = _get_semaphore_injection()
    val_arch = _get_semaphore_injection()
    fusion_shared_val_arch = FusionSharedValArch(obs, val_arch)
    fusion_shared_val.putChild(path=b'iter-arch', child=fusion_shared_val_arch)

    val_reward = _get_semaphore_injection()
    shared_val_reward = FusionSharedValReward(val_reward)
    fusion_shared_val.putChild(path=b'reward', child=shared_val_reward)

    shared_val_checkpoint = FusionSharedValCheckpoint(checkpoint)
    fusion_shared_val.putChild(path=b'checkpoint', child=shared_val_checkpoint)

    return obs, val_arch, val_reward


def _setup_shared_train(fusion):
    """Setup shared train pages, and return checkpoint, iteration
    architectures, and minibatches per epoch resources.
    """
    shared_train = twisted.web.resource.Resource()
    fusion.putChild(path=b'shared-train', child=shared_train)

    iter_arch = _get_semaphore_injection()
    minibatches_per_epoch = _get_semaphore_injection()
    train_ex_features = _get_semaphore_injection()
    shared_iter_arch = FusionSharedIterArch(iter_arch,
                                            minibatches_per_epoch,
                                            train_ex_features)
    shared_train.putChild(path=b'iter-arch', child=shared_iter_arch)

    checkpoint = _get_semaphore_injection()
    shared_send_checkpoint = FusionSharedSendCheckpoint(checkpoint)
    shared_train.putChild(path=b'checkpoint', child=shared_send_checkpoint)

    return (checkpoint,
            iter_arch,
            minibatches_per_epoch,
            train_ex_features)


def _setup_controller_train(fusion,
                            val_arch,
                            val_reward,
                            iter_arch,
                            minibatches_per_epoch,
                            train_ex_features,
                            obs):
    """Setup controller training pages."""
    controller_train = twisted.web.resource.Resource()
    fusion.putChild(path=b'controller-train', child=controller_train)

    controller_get_train_features = FusionControllerGetTrainFeatures(
        train_ex_features)
    controller_train.putChild(path=b'get-train-features',
                              child=controller_get_train_features)

    controller_get_obs = FusionControllerGetObs(obs)
    controller_train.putChild(path=b'get-obs', child=controller_get_obs)

    controller_iter_arch = FusionControllerIterArch(iter_arch)
    controller_train.putChild(path=b'iter-arch', child=controller_iter_arch)

    controller_step = FusionControllerStepReward(val_arch, val_reward)
    controller_train.putChild(path=b'step-reward', child=controller_step)

    controller_minibatches = FusionControllerMinibatches(minibatches_per_epoch)
    controller_train.putChild(path=b'minibatches_per_epoch',
                              child=controller_minibatches)


@click.command()
@click.option('--port', type=int, default=None, help='Port to listen on.')
def parameter_server(port):
    """Runs a parameter server that stores and updates a set of parameters."""
    twisted.python.log.startLogging(sys.stdout)

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

    # TODO(brendan): Split out the fusion operator stuff from the parameter
    # server...
    fusion = twisted.web.resource.Resource()
    root.putChild(path=b'fusion', child=fusion)

    (checkpoint,
     iter_arch,
     minibatches_per_epoch,
     train_ex_features) = _setup_shared_train(fusion)

    obs, val_arch, val_reward = _setup_shared_val(fusion, checkpoint)

    _setup_controller_train(fusion,
                            val_arch,
                            val_reward,
                            iter_arch,
                            minibatches_per_epoch,
                            train_ex_features,
                            obs)

    site = twisted.web.server.Site(resource=root)

    endpoint = twisted.internet.endpoints.TCP4ServerEndpoint(
        reactor=twisted.internet.reactor, port=port)
    endpoint.listen(site)

    twisted.internet.reactor.run()
