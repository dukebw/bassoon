"""A parameter server meant to be per-experiment, which receives an initial set
of parameters, then applies updates to those parameters, returning the updated
parameters.

There should be two functions exposed:

initialize(): takes a set of parameter values as the initial parameters.

reset(): reset the server to its uninitialized state.

update(): updates the stored set of parameters atomically, and returns the
updated parameters to the client.
"""
import twisted.web.resource
import twisted.web.server
import twisted.internet.endpoints
import twisted.internet.reactor


HTTP_STATUS_ALREADY_INIT = 299


class Initialize(twisted.web.resource.Resource):
    isLeaf = True

    def render_GET(self, request):
        return '<!DOCTYPE html><meta charset="utf-8"/><html>你好，世界!</html>'.encode('utf-8')

    def render_POST(self, request):
        print(len(request.content.read()))
        return '<html>Hello, world!</html>'.encode('utf-8')


def parameter_server():
    root = twisted.web.resource.Resource()
    root.putChild(path=b'init', child=Initialize())

    site = twisted.web.server.Site(resource=root)

    endpoint = twisted.internet.endpoints.TCP4ServerEndpoint(
        reactor=twisted.internet.reactor, port=8880)
    endpoint.listen(site)

    twisted.internet.reactor.run()
