"""A module for datatypes."""


class Struct:
    """A convenient struct-like class.

    Source: Python Cookbook, Beazley and Jones. 3rd Edition. 8.11. Simplifying
        the Initialization of Data Structures.

    Usage:
        ```python
        class Stock(Structure):
            _fields = ['name', 'shares', 'price']

        s1 = Stock('ACME', 50, 91.1)
        s2 = Stock('ACME', 50, price=91.1)
        s3 = Stock('ACME', shares=50, price=91.1)
        ```
    """
    _fields = []

    def __init__(self, *args, **kwargs):
        if len(args) > len(self._fields):
            raise TypeError('Expected {} arguments'.format(len(self._fields)))

        # Set all of the positional arguments
        for name, value in zip(self._fields, args):
            setattr(self, name, value)

        # Set the remaining keyword arguments
        for name in self._fields[len(args):]:
            setattr(self, name, kwargs.pop(name))

        # Check for any remaining unknown arguments
        if kwargs:
            raise TypeError('Invalid argument(s): {}'.format(','.join(kwargs)))
