"""Client library code for bassoon."""
import io

import twisted


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


def post_params(agent, params_val, page):
    """Make a POST request to localhost port 8880 with `params_val` as the
    request body.
    """
    body = twisted.web.client.FileBodyProducer(
        inputFile=io.BytesIO(params_val.tobytes()))

    return post_body(agent, page, body)
