
from collections import OrderedDict
from .base import AbstractMetaFeature


class MetaFeature(AbstractMetaFeature):
    def __init__(self):
        super(MetaFeature, self).__init__()
        self.type_ = "METAFEATURE"


class HelperFunction(AbstractMetaFeature):
    def __init__(self):
        super(HelperFunction, self).__init__()
        self.type_ = "HELPERFUNCTION"


class MetafeatureFunctions(object):
    def __init__(self):
        self.functions = OrderedDict()
        self.dependencies = OrderedDict()
        self.values = OrderedDict()

    def clear(self):
        self.values = OrderedDict()

    def __iter__(self):
        return self.functions.__iter__()

    def __getitem__(self, item):
        return self.functions.__getitem__(item)

    def __setitem__(self, key, value):
        return self.functions.__setitem__(key, value)

    def __delitem__(self, key):
        return self.functions.__delitem__(key)

    def __contains__(self, item):
        return self.functions.__contains__(item)

    def get_value(self, key):
        return self.values[key].value

    def set_value(self, key, item):
        self.values[key] = item

    def is_calculated(self, key):
        """Return if a helper function has already been executed.

        Necessary as get_value() can return None if the helper function hasn't
        been executed or if it returned None."""
        return key in self.values

    def get_dependency(self, name):
        """Return the dependency of metafeature "name".
        """
        return self.dependencies.get(name)

    def define(self, name, dependency=None):
        """Decorator for adding metafeature functions to a "dictionary" of
        metafeatures. This behaves like a function decorating a function,
        not a class decorating a function"""
        def wrapper(metafeature_class):
            instance = metafeature_class()
            self.__setitem__(name, instance)
            self.dependencies[name] = dependency
            return instance
        return wrapper


# TODO Allow multiple dependencies for a metafeature
# TODO Add HelperFunction as an object
class HelperFunctions(object):
    def __init__(self):
        self.functions = OrderedDict()
        self.values = OrderedDict()

    def clear(self):
        self.values = OrderedDict()
        self.computation_time = OrderedDict()

    def __iter__(self):
        return self.functions.__iter__()

    def __getitem__(self, item):
        return self.functions.__getitem__(item)

    def __setitem__(self, key, value):
        return self.functions.__setitem__(key, value)

    def __delitem__(self, key):
        return self.functions.__delitem__(key)

    def __contains__(self, item):
        return self.functions.__contains__(item)

    def is_calculated(self, key):
        """Return if a helper function has already been executed.

        Necessary as get_value() can return None if the helper function hasn't
        been executed or if it returned None."""
        return key in self.values

    def get_value(self, key):
        return self.values.get(key).value

    def set_value(self, key, item):
        self.values[key] = item

    def define(self, name):
        """Decorator for adding helper functions to a "dictionary".
        This behaves like a function decorating a function,
        not a class decorating a function"""
        def wrapper(metafeature_class):
            instance = metafeature_class()
            self.__setitem__(name, instance)
            return instance
        return wrapper
